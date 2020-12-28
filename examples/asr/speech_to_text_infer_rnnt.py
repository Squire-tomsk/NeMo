# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script serves three goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch ASR inference
    (3) Serves as CI test for pre-trained checkpoint
"""

from argparse import ArgumentParser

import torch
import json
from tqdm.auto import tqdm

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.metrics.rnnt_wer import RNNTWER
from nemo.collections.asr.models import EncDecRNNTModel
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


can_gpu = torch.cuda.is_available()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, default="QuartzNet15x5Base-En", required=True, help="Pass: 'QuartzNet15x5Base-En'",
    )
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--wer_tolerance", type=float, default=1.0, help="used by test")
    parser.add_argument(
        "--normalize_text", default=True, type=bool, help="Normalize transcripts or not. Set to False for non-English."
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--output_json", type=str)
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = EncDecRNNTModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = EncDecRNNTModel.from_pretrained(model_name=args.asr_model)
    asr_model.setup_test_data(
        test_data_config={
            'sample_rate': args.sample_rate,
            'manifest_filepath': args.dataset,
            'labels': asr_model.joint.vocabulary,
            'batch_size': args.batch_size,
            'normalize_transcripts': args.normalize_text,
        }
    )
    if can_gpu:
        asr_model = asr_model.cuda()
    asr_model.eval()
    labels_map = dict([(i, asr_model.joint.vocabulary[i]) for i in range(len(asr_model.joint.vocabulary))])
    wer = RNNTWER(asr_model.decoding)
    hypotheses = []
    references = []
    for test_batch in tqdm(asr_model.test_dataloader()):
        if can_gpu:
            test_batch = [x.cuda() for x in test_batch]
        with autocast():
            encoded, encoded_len = asr_model(
                input_signal=test_batch[0], input_signal_length=test_batch[1]
            )
            hypotheses += wer.decoding.rnnt_decoder_predictions_tensor(encoded, encoded_len)[0]
        for batch_ind in range(test_batch[0].shape[0]):
            reference = ''.join([labels_map[c] for c in test_batch[2][batch_ind].cpu().detach().numpy()])
            references.append(reference)
        print(f'REF: {references[-1]} HYP: {hypotheses[-1]}')
        del test_batch
    wer_value = word_error_rate(hypotheses=hypotheses, references=references)
    logging.info(f'Got WER of {wer_value}. Tolerance was {args.wer_tolerance}')
    if args.output_json is not None:
        logging.info(f'Saving result to {args.output_json}')
        result = []
        for hyp, ref in zip(hypotheses, references):
            result.append({'hyp': hyp, 'ref': ref})
        json.dump(result, open(args.output_json, 'w'), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
