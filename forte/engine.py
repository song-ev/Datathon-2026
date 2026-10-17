"""Local video inference. Never calls a model API or downloads at runtime."""
import base64
import io
import json
import math
import os
import time
from pathlib import Path

import av
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

RULEBOOK = json.loads(Path(__file__).with_name('rules.json').read_text(encoding='utf8'))
RULES = {r['id']: r for r in RULEBOOK['rules']}
MODEL_PATH = os.getenv('MODEL_PATH', '/home/ubuntu/forte-datathon/models/Qwen3-VL-4B-Instruct')


def parse_json(text):
    """Accept fenced JSON, but never invent a successful empty result on parse errors."""
    start = text.find('{')
    if start < 0:
        raise ValueError('Model did not return JSON')
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj.get('events'), list):
        raise ValueError('Missing events list')
    return obj


def probe(path):
    with av.open(str(path)) as c:
        if not c.streams.video:
            raise ValueError('В файле нет видеопотока')
        s = c.streams.video[0]
        duration = float(s.duration * s.time_base) if s.duration else float(c.duration or 0)/av.time_base
        if not 0 < duration <= 180:
            raise ValueError('Поддерживаются видео длительностью до 180 секунд')
        if s.width * s.height > 3840*2160:
            raise ValueError('Разрешение превышает 4K')
        return duration


def sample(path, step=1.0):
    result, target = [], 0.0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = 'AUTO'
        for frame in c.decode(s):
            t = float(frame.time or 0)
            if t+0.001 >= target:
                image = frame.to_image()
                image.thumbnail((448, 672))
                result.append((round(t, 3), image))
                target += step
    if not result:
        raise ValueError('Не удалось декодировать кадры')
    return result


def nearest(frames, timestamp):
    return min(frames, key=lambda item: abs(item[0]-timestamp))


def evidence(path, timestamps):
    """Decode original source frames, not generated or resized model inputs."""
    pending = sorted(set(max(0.0, float(t)) for t in timestamps))
    result = []
    with av.open(str(path)) as c:
        for frame in c.decode(video=0):
            t = float(frame.time or 0)
            if pending and t+0.001 >= pending[0]:
                b = io.BytesIO()
                frame.to_image().save(b, format='JPEG', quality=88)
                result.append({'timestamp_sec': round(t, 3), 'image_base64': base64.b64encode(b.getvalue()).decode()})
                while pending and t+0.001 >= pending[0]:
                    pending.pop(0)
            if not pending:
                break
    return result[:3]


class UnsupportedScene(ValueError):
    pass


class Engine:
    def __init__(self):
        torch.set_num_threads(4)
        self.processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
            attn_implementation='sdpa', local_files_only=True).eval()
        self.threshold = float(os.getenv('EVENT_THRESHOLD', '0.65'))
        self.prompt = (
            'You audit silent bank cashier video. Analyze only visible actions, never identity, age, gender, '
            'emotion, or intentions. Images are successive frames with absolute timestamps. '
            'Find concrete violations of these rules; apply all requirements and exceptions. '
            'Do not treat missing evidence as a violation. Cash on desk during current transaction, '
            'using a phone, standing up, machine counting, and a single cashier are not by themselves violations. '
            'Output ONLY JSON: {"events":[{"rule_id":"...","time_sec":12.0,"confidence":0.8,'
            '"observation":"short factual Russian description of the visible action and context"}]}. '
            'Use events=[] if none. Confidence is your evidence strength, not probability. '
            'At most 2 events. Timestamp must be one of the supplied frames. '
            'Rules:\n' + json.dumps(RULEBOOK['rules'], ensure_ascii=False)
        )

    def infer(self, frames, extra='', tokens=220, prompt=None):
        content = []
        for t, im in frames:
            content.extend([{'type':'text','text':f't={t:.2f}s'}, {'type':'image','image':im}])
        content.append({'type':'text','text':(self.prompt if prompt is None else prompt)+'\n'+extra})
        inputs = self.processor.apply_chat_template([{'role':'user','content':content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=tokens, do_sample=False)
            text = self.processor.batch_decode(output[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0]
            return parse_json(text)
        finally:
            del inputs

    def analyze_rules(self, path, progress=None, include_evidence=True, time_limit=265):
        started = time.monotonic()
        duration = probe(path)
        frames = sample(path)
        starts = list(range(0, max(1, math.ceil(duration)-2), 10))
        candidates, failures = [], 0
        for n, begin in enumerate(starts):
            if time.monotonic()-started > time_limit-35:
                raise TimeoutError('Недостаточно времени для анализа всего видео')
            end = min(duration-0.05, begin+12)
            selected = [nearest(frames, begin+(end-begin)*i/7) for i in range(8)]
            try:
                answer = self.infer(selected)
            except (ValueError, json.JSONDecodeError):
                failures += 1
                answer = self.infer(selected, 'Return valid short JSON. One event maximum, or events=[].', tokens=180)
            for e in answer['events']:
                try:
                    rid, t, conf = e['rule_id'], float(e['time_sec']), float(e['confidence'])
                    if rid not in RULES or not math.isfinite(t) or not math.isfinite(conf):
                        continue
                    if not begin-1 <= t <= end+1 or not 0 <= conf <= 1:
                        continue
                    text = str(e.get('observation','')).strip()[:1200]
                    if not text:
                        continue
                    candidates.append(dict(rule_id=rid,time_sec=nearest(frames,t)[0],confidence=conf,observation=text))
                except (KeyError, TypeError, ValueError):
                    continue
            if progress:
                progress(round((n+1)/len(starts)*85), 'Анализ видеосегментов')
        # Merge temporal overlap without merging distinct distant events.
        merged = []
        for e in sorted(candidates, key=lambda x:-x['confidence']):
            if not any(m['rule_id']==e['rule_id'] and abs(m['time_sec']-e['time_sec'])<9 for m in merged):
                merged.append(e)
        violations = []
        for e in sorted(merged, key=lambda x:x['time_sec']):
            if e['confidence'] < self.threshold:
                continue
            rule = RULES[e['rule_id']]
            t = min(max(0,e['time_sec']),duration-0.05)
            refs = {'document':RULEBOOK['document'], 'paragraph':rule['paragraph'],
                    'source_url':RULEBOOK['source_url'], 'rule_id':e['rule_id'], 'summary':rule['summary']}
            violations.append({'type':rule['title'], 'timecode_sec':round(t,3),
                'description':e['observation']+' Основание: постановление НБРК № 231, п. '+rule['paragraph']+'. '+rule['summary'],
                'evidence_frames':evidence(path,[max(0,t-1),t,min(duration-0.1,t+1)]) if include_evidence else [],
                'normative_reference':refs, 'confidence':e['confidence']})
        return {'verdict':'violation' if violations else 'no_violation', 'violations':violations,
            'analysis':{'duration_sec':round(duration,3), 'processing_sec':round(time.monotonic()-started,3),
                'segments':len(starts), 'model':'Qwen3-VL-4B-Instruct', 'rulebook_version':RULEBOOK['version'],
                'json_retries':failures, 'confidence_is_calibrated':False,
                'limitations':'Автоматическая проверка видимых действий; внекадровые действия и внутренние регламенты не устанавливаются.'},
            '_candidates':merged}

    def analyze(self, path, progress=None, include_evidence=True, time_limit=265):
        detector_path = Path(os.getenv('FORTE_HOME', '/home/ubuntu/forte-datathon'))/'models/video-detector.joblib'
        if os.getenv('DETECTOR_MODE', 'trained') == 'rules' or not detector_path.exists():
            return self.analyze_rules(path, progress, include_evidence, time_limit)
        import joblib
        import numpy as np
        from forte.features import extract, temporal
        started = time.monotonic()
        duration = probe(path)
        if not hasattr(self, 'video_detector'):
            self.video_detector = joblib.load(detector_path)
            self.event_detector = joblib.load(detector_path.with_name('event-detector.joblib'))
        scene_frames = sample(path, step=max(1.0,duration/3))[:3]
        scene = self.infer(scene_frames, tokens=70, prompt='Decide whether these frames show a real indoor cashier/cash-handling workstation or bank service counter. Do not identify people. A real office counter with a worker is sufficient; cash need not be visible. Return ONLY JSON {"events":[{"cashier_scene":true}]} if yes. For cartoons, geometric shapes, blank images, outdoor scenes or unrelated content return {"events":[]}.')
        if not any(e.get('cashier_scene') is True for e in scene['events'] if isinstance(e,dict)):
            raise UnsupportedScene('Видео не содержит поддерживаемую сцену кассового рабочего места')
        times, vectors = extract(self, path)
        difference = np.abs(np.diff(vectors, axis=0)).mean(0) if len(vectors)>1 else np.zeros_like(vectors[0])
        aggregate = np.concatenate([vectors.mean(0), vectors.std(0), difference])[None]
        score = float(self.video_detector['pipeline'].predict_proba(aggregate)[0,1])
        positive = score >= self.video_detector['threshold']
        if progress:
            progress(20, 'Классификация видео завершена')
        result = {'verdict':'no_violation', 'violations':[], '_candidates':[], 'analysis':{'duration_sec':round(duration,3),'segments':0}}
        if positive:
            result = self.analyze_rules(path, progress, include_evidence, time_limit-(time.monotonic()-started))
            if not result['violations']:
                frame_scores = self.event_detector['pipeline'].predict_proba(temporal(vectors))[:,1]
                t = float(times[int(np.argmax(frame_scores))])
                result['violations'] = [{'type':'Отклонение в кассовой операции — тип требует проверки',
                    'timecode_sec':round(t,3),
                    'description':'Обученный на разметке видеоклассификатор обнаружил признаки нарушения. Конкретное действие и применимый пункт нормы не подтверждены. Время указывает наиболее подозрительный фрагмент; требуется проверка специалистом.',
                    'evidence_frames':evidence(path,[max(0,t-1),t,min(duration-0.1,t+1)]) if include_evidence else [],
                    'normative_reference':None, 'confidence':score, 'requires_review':True}]
            result['verdict'] = 'violation'
        result['_candidates'] = [{'time_sec':v['timecode_sec'],'confidence':1.0} for v in result['violations']]
        result['analysis'].update(processing_sec=round(time.monotonic()-started,3),
            model='Qwen3-VL-4B-Instruct + supervised video classifier',
            detector_score=score, detector_threshold=self.video_detector['threshold'],
            classifier_training_groups=24, confidence_is_calibrated=False,
            rulebook_version=RULEBOOK['version'],
            limitations='Малый обучающий набор и один ракурс. Неподтверждённый тип и нормативное основание явно отмечаются; решение требует проверки специалистом.')
        return result
