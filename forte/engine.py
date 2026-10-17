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

    def infer(self, frames, extra='', tokens=220):
        content = []
        for t, im in frames:
            content.extend([{'type':'text','text':f't={t:.2f}s'}, {'type':'image','image':im}])
        content.append({'type':'text','text':self.prompt+'\n'+extra})
        inputs = self.processor.apply_chat_template([{'role':'user','content':content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
        try:
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=tokens, do_sample=False)
            text = self.processor.batch_decode(output[:, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)[0]
            return parse_json(text)
        finally:
            del inputs

    def analyze(self, path, progress=None, include_evidence=True, time_limit=265):
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
