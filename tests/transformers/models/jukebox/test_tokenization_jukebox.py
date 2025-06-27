# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import unittest

from mindnlp.transformers import JukeboxTokenizer
from mindnlp.utils.testing_utils import require_mindspore


class JukeboxTokenizationTest(unittest.TestCase):
    tokenizer_class = JukeboxTokenizer
    metas = {
        "artist": "Zac Brown Band",
        "genres": "Country",
        "lyrics": """I met a traveller from an antique land,
        Who said "Two vast and trunkless legs of stone
        Stand in the desert. . . . Near them, on the sand,
        Half sunk a shattered visage lies, whose frown,
        And wrinkled lip, and sneer of cold command,
        Tell that its sculptor well those passions read
        Which yet survive, stamped on these lifeless things,
        The hand that mocked them, and the heart that fed;
        And on the pedestal, these words appear:
        My name is Ozymandias, King of Kings;
        Look on my Works, ye Mighty, and despair!
        Nothing beside remains. Round the decay
        Of that colossal Wreck, boundless and bare
        The lone and level sands stretch far away
        """,
    }

    @require_mindspore
    def test_1b_lyrics_tokenizer(self):
        """
        how to run the same test with openAI
        ...
        """
        import mindspore
        import numpy as np

        tokenizer = JukeboxTokenizer.from_pretrained("openai/jukebox-1b-lyrics")
        tokens = tokenizer(**self.metas)["input_ids"]
        # fmt: off
        EXPECTED_OUTPUT = [
            mindspore.tensor([[
                0, 0, 0, 7169, 507, 9, 76, 39, 31, 46, 76, 27,
                76, 46, 44, 27, 48, 31, 38, 38, 31, 44, 76, 32,
                44, 41, 39, 76, 27, 40, 76, 27, 40, 46, 35, 43,
                47, 31, 76, 38, 27, 40, 30, 64, 78, 76, 76, 76,
                76, 76, 76, 76, 76, 23, 34, 41, 76, 45, 27, 35,
                30, 76, 71, 20, 49, 41, 76, 48, 27, 45, 46, 76,
                27, 40, 30, 76, 46, 44, 47, 40, 37, 38, 31, 45,
                45, 76, 38, 31, 33, 45, 76, 41, 32, 76, 45, 46,
                41, 40, 31, 78, 76, 76, 76, 76, 76, 76, 76, 76,
                19, 46, 27, 40, 30, 76, 35, 40, 76, 46, 34, 31,
                76, 30, 31, 45, 31, 44, 46, 63, 76, 63, 76, 63,
                76, 63, 76, 14, 31, 27, 44, 76, 46, 34, 31, 39,
                64, 76, 41, 40, 76, 46, 34, 31, 76, 45, 27, 40,
                30, 64, 78, 76, 76, 76, 76, 76, 76, 76, 76, 8,
                27, 38, 32, 76, 45, 47, 40, 37, 76, 27, 76, 45,
                34, 27, 46, 46, 31, 44, 31, 30, 76, 48, 35, 45,
                27, 33, 31, 76, 38, 35, 31, 45, 64, 76, 49, 34,
                41, 45, 31, 76, 32, 44, 41, 49, 40, 64, 78, 76,
                76, 76, 76, 76, 76, 76, 76, 1, 40, 30, 76, 49,
                44, 35, 40, 37, 38, 31, 30, 76, 38, 35, 42, 64,
                76, 27, 40, 30, 76, 45, 40, 31, 31, 44, 76, 41,
                32, 76, 29, 41, 38, 30, 76, 29, 41, 39, 39, 27,
                40, 30, 64, 78, 76, 76, 76, 76, 76, 76, 76, 76,
                20, 31, 38, 38, 76, 46, 34, 27, 46, 76, 35, 46,
                45, 76, 45, 29, 47, 38, 42, 46, 41, 44, 76, 49,
                31, 38, 38, 76, 46, 34, 41, 45, 31, 76, 42, 27,
                45, 45, 35, 41, 40, 45, 76, 44, 31, 27, 30, 78,
                76, 76, 76, 76, 76, 76, 76, 76, 23, 34, 35, 29,
                34, 76, 51, 31, 46, 76, 45, 47, 44, 48, 35, 48,
                31, 64, 76, 45, 46, 27, 39, 42, 31, 30, 76, 41,
                40, 76, 46, 34, 31, 45, 31, 76, 38, 35, 32, 31,
                38, 31, 45, 45, 76, 46, 34, 35, 40, 33, 45, 64,
                78, 76, 76, 76, 76, 76, 76, 76, 76, 20, 34, 31,
                76, 34, 27, 40, 30, 76, 46, 34, 27, 46, 76, 39,
                41, 29, 37, 31, 30, 76, 46, 34, 31, 39, 64, 76,
                27, 40, 30, 76, 46, 34, 31, 76, 34, 31, 27, 44,
                46, 76, 46, 34, 27, 46, 76, 32, 31, 30, 66, 78,
                76, 76, 76, 76, 76, 76, 76, 76, 1, 40, 30, 76,
                41, 40, 76, 46, 34, 31, 76, 42, 31, 30, 31, 45,
                46, 27, 38, 64, 76, 46, 34, 31, 45, 31, 76, 49,
                41, 44, 30, 45, 76, 27, 42, 42, 31, 27, 44, 65,
                78, 76, 76, 76, 76, 76, 76, 76, 76, 13, 51, 76,
                40, 27, 39, 31, 76, 35, 45, 76, 15, 52, 51, 39,
                27, 40, 30, 35, 27, 45, 64, 76, 11, 35, 40, 33,
                76, 41, 32, 76, 11, 35, 40, 33, 45, 66, 78, 76,
                76, 76, 76, 76, 76, 76, 76, 12, 41, 41, 37, 76,
                41, 40, 76, 39, 51, 76, 23, 41, 44, 37, 45, 64,
                76, 51, 31, 76, 13, 35, 33, 34, 46, 51, 64, 76,
                27, 40, 30, 76, 30, 31, 45, 42, 27, 35, 44, 67,
                78, 76, 76, 76, 76, 76, 76, 76, 76, 14, 41, 46,
                34, 35, 40, 33, 76, 28, 31, 45, 35, 30, 31, 76,
                44, 31, 39, 27, 35, 40, 45, 63, 76, 18, 41, 47,
                40, 30, 76, 46, 34, 31, 76, 30, 31, 29, 27, 51,
                78, 76, 76, 76, 76, 76, 76, 76, 76, 15, 32, 76,
                46, 34, 27, 46, 76, 29, 41, 38, 41, 45, 45, 27,
                38, 76, 23, 44, 31, 29, 37, 64, 76, 28, 41, 47,
                40, 30, 38, 31, 45, 45, 76, 27, 40, 30, 76, 28,
                27, 44, 31, 78, 76, 76, 76, 76, 76, 76, 76, 76,
                20, 34, 31, 76, 38, 41, 40, 31, 76, 27, 40, 30,
                76, 38, 31, 48, 31, 38, 76, 45, 27, 40, 30, 45,
                76, 45, 46, 44, 31, 46, 29, 34, 76, 32, 27, 44,
                76, 27, 49, 27, 51, 78, 76, 76, 76, 76, 76, 76,
                76, 76]]),
            mindspore.tensor([[0, 0, 0, 1069, 11]]),
            mindspore.tensor([[0, 0, 0, 1069, 11]]),
        ]
        # fmt: on
        self.assertTrue(np.allclose(tokens[0].asnumpy(), EXPECTED_OUTPUT[0].asnumpy()))
        self.assertTrue(np.allclose(tokens[1].asnumpy(), EXPECTED_OUTPUT[1].asnumpy()))
        self.assertTrue(np.allclose(tokens[2].asnumpy(), EXPECTED_OUTPUT[2].asnumpy()))

    @require_mindspore
    def test_5b_lyrics_tokenizer(self):
        """
        The outputs are similar that open AI but do not have the same format as this one is adapted to the HF integration.
        """
        import mindspore
        import numpy as np

        tokenizer = JukeboxTokenizer.from_pretrained("openai/jukebox-5b-lyrics")
        tokens = tokenizer(**self.metas)["input_ids"]
        # fmt: off
        EXPECTED_OUTPUT = [
            mindspore.tensor([[
                0, 0, 0, 1069, 11, -1, -1, -1, -1, 9, 77, 39,
                31, 46, 77, 27, 77, 46, 44, 27, 48, 31, 38, 38,
                31, 44, 77, 32, 44, 41, 39, 77, 27, 40, 77, 27,
                40, 46, 35, 43, 47, 31, 77, 38, 27, 40, 30, 64,
                79, 77, 77, 77, 77, 77, 77, 77, 77, 23, 34, 41,
                77, 45, 27, 35, 30, 77, 72, 20, 49, 41, 77, 48,
                27, 45, 46, 77, 27, 40, 30, 77, 46, 44, 47, 40,
                37, 38, 31, 45, 45, 77, 38, 31, 33, 45, 77, 41,
                32, 77, 45, 46, 41, 40, 31, 79, 77, 77, 77, 77,
                77, 77, 77, 77, 19, 46, 27, 40, 30, 77, 35, 40,
                77, 46, 34, 31, 77, 30, 31, 45, 31, 44, 46, 63,
                77, 63, 77, 63, 77, 63, 77, 14, 31, 27, 44, 77,
                46, 34, 31, 39, 64, 77, 41, 40, 77, 46, 34, 31,
                77, 45, 27, 40, 30, 64, 79, 77, 77, 77, 77, 77,
                77, 77, 77, 8, 27, 38, 32, 77, 45, 47, 40, 37,
                77, 27, 77, 45, 34, 27, 46, 46, 31, 44, 31, 30,
                77, 48, 35, 45, 27, 33, 31, 77, 38, 35, 31, 45,
                64, 77, 49, 34, 41, 45, 31, 77, 32, 44, 41, 49,
                40, 64, 79, 77, 77, 77, 77, 77, 77, 77, 77, 1,
                40, 30, 77, 49, 44, 35, 40, 37, 38, 31, 30, 77,
                38, 35, 42, 64, 77, 27, 40, 30, 77, 45, 40, 31,
                31, 44, 77, 41, 32, 77, 29, 41, 38, 30, 77, 29,
                41, 39, 39, 27, 40, 30, 64, 79, 77, 77, 77, 77,
                77, 77, 77, 77, 20, 31, 38, 38, 77, 46, 34, 27,
                46, 77, 35, 46, 45, 77, 45, 29, 47, 38, 42, 46,
                41, 44, 77, 49, 31, 38, 38, 77, 46, 34, 41, 45,
                31, 77, 42, 27, 45, 45, 35, 41, 40, 45, 77, 44,
                31, 27, 30, 79, 77, 77, 77, 77, 77, 77, 77, 77,
                23, 34, 35, 29, 34, 77, 51, 31, 46, 77, 45, 47,
                44, 48, 35, 48, 31, 64, 77, 45, 46, 27, 39, 42,
                31, 30, 77, 41, 40, 77, 46, 34, 31, 45, 31, 77,
                38, 35, 32, 31, 38, 31, 45, 45, 77, 46, 34, 35,
                40, 33, 45, 64, 79, 77, 77, 77, 77, 77, 77, 77,
                77, 20, 34, 31, 77, 34, 27, 40, 30, 77, 46, 34,
                27, 46, 77, 39, 41, 29, 37, 31, 30, 77, 46, 34,
                31, 39, 64, 77, 27, 40, 30, 77, 46, 34, 31, 77,
                34, 31, 27, 44, 46, 77, 46, 34, 27, 46, 77, 32,
                31, 30, 66, 79, 77, 77, 77, 77, 77, 77, 77, 77,
                1, 40, 30, 77, 41, 40, 77, 46, 34, 31, 77, 42,
                31, 30, 31, 45, 46, 27, 38, 64, 77, 46, 34, 31,
                45, 31, 77, 49, 41, 44, 30, 45, 77, 27, 42, 42,
                31, 27, 44, 65, 79, 77, 77, 77, 77, 77, 77, 77,
                77, 13, 51, 77, 40, 27, 39, 31, 77, 35, 45, 77,
                15, 52, 51, 39, 27, 40, 30, 35, 27, 45, 64, 77,
                11, 35, 40, 33, 77, 41, 32, 77, 11, 35, 40, 33,
                45, 66, 79, 77, 77, 77, 77, 77, 77, 77, 77, 12,
                41, 41, 37, 77, 41, 40, 77, 39, 51, 77, 23, 41,
                44, 37, 45, 64, 77, 51, 31, 77, 13, 35, 33, 34,
                46, 51, 64, 77, 27, 40, 30, 77, 30, 31, 45, 42,
                27, 35, 44, 67, 79, 77, 77, 77, 77, 77, 77, 77,
                77, 14, 41, 46, 34, 35, 40, 33, 77, 28, 31, 45,
                35, 30, 31, 77, 44, 31, 39, 27, 35, 40, 45, 63,
                77, 18, 41, 47, 40, 30, 77, 46, 34, 31, 77, 30,
                31, 29, 27, 51, 79, 77, 77, 77, 77, 77, 77, 77,
                77, 15, 32, 77, 46, 34, 27, 46, 77, 29, 41, 38,
                41, 45, 45, 27, 38, 77, 23, 44, 31, 29, 37, 64,
                77, 28, 41, 47, 40, 30, 38, 31, 45, 45, 77, 27,
                40, 30, 77, 28, 27, 44, 31, 79, 77, 77, 77, 77,
                77, 77, 77, 77, 20, 34, 31, 77, 38, 41, 40, 31,
                77, 27, 40, 30, 77, 38, 31, 48, 31, 38, 77, 45,
                27, 40, 30, 45, 77, 45, 46, 44, 31, 46, 29, 34,
                77, 32, 27, 44, 77, 27, 49, 27, 51, 79, 77, 77,
                77, 77, 77, 77, 77, 77]]),
            mindspore.tensor([[0, 0, 0, 1069, 11, -1, -1, -1, -1]]),
            mindspore.tensor([[0, 0, 0, 1069, 11, -1, -1, -1, -1]]),
        ]
        # fmt: on
        self.assertTrue(np.allclose(tokens[0].asnumpy(), EXPECTED_OUTPUT[0].asnumpy()))
        self.assertTrue(np.allclose(tokens[1].asnumpy(), EXPECTED_OUTPUT[1].asnumpy()))
        self.assertTrue(np.allclose(tokens[2].asnumpy(), EXPECTED_OUTPUT[2].asnumpy())) 