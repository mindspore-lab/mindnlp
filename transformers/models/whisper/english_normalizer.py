# Copyright 2022 The OpenAI team and The HuggingFace Team. All rights reserved.
# Most of the code is copy pasted from the original whisper repository
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
# ============================================================================
"""English normalizer."""
import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union

import regex


# non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space, and drop any diacritics (category 'Mn' and some
    manual mappings)
    """
    def replace_character(char):
        if char in keep:
            return char
        if char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]

        if unicodedata.category(char) == "Mn":
            return ""

        if unicodedata.category(char)[0] in "MSP":
            return " "

        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space, keeping diacritics
    """
    return "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))


class BasicTextNormalizer:

    """
    The BasicTextNormalizer class is responsible for normalizing text by removing symbols and diacritics (if specified) a
    nd splitting letters (if specified).
    
    Attributes:
        remove_diacritics (bool): A flag indicating whether to remove diacritics from the text. Default is False.
        split_letters (bool): A flag indicating whether to split letters in the text. Default is False.

    Methods:
        __call__: Normalizes the input text by converting it to lowercase, removing HTML tags and content within
            parentheses, applying symbol and diacritic removal, and optionally splitting letters if the split_letters
            flag is True. Returns the normalized text.

    Example:
        ```python
        >>> normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
        >>> normalized_text = normalizer('Hello, World!')
        >>> print(normalized_text)  # Output: 'hello world'
        ```

    """
    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False):
        """
        Initializes an instance of the BasicTextNormalizer class.

        Args:
            self: The instance of the BasicTextNormalizer class.
            remove_diacritics (bool): A boolean flag indicating whether diacritics should be removed from the text.
                Defaults to False.
            split_letters (bool): A boolean flag indicating whether letters should be split. Defaults to False.

        Returns:
            None.

        Raises:
            None.
        """
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        self.split_letters = split_letters

    def __call__(self, s: str):
        """
        This method normalizes the input text by converting it to lowercase, removing certain patterns and characters,
        and optionally splitting the text into individual letters.

        Args:
            self (object): The instance of the BasicTextNormalizer class.
            s (str): The input text to be normalized. It should be a valid string.

        Returns:
            None: This method does not return any value. The input text 's' is modified in place.

        Raises:
            None.
        """
        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s


class EnglishNumberNormalizer:
    """
    Convert any spelled-out numbers into arabic numbers, while handling:

    - remove any commas
    - keep the suffixes such as: `1960s`, `274th`, `32nd`, etc.
    - spell out currency symbols after the number. e.g. `$20 million` -> `20000000 dollars`
    - spell out `one` and `ones`
    - interpret successive single-digit numbers as nominal: `one oh one` -> `101`
    """
    def __init__(self):
        """
        Initializes an instance of the EnglishNumberNormalizer class.

        Args:
            self: The instance of the class.

        Returns:
            None.

        Raises:
            None.

        This method initializes the instance of the EnglishNumberNormalizer class.
        It sets up various dictionaries and mappings that are used for normalizing English numbers.
        The dictionaries include:

        - 'zeros': A set of strings representing the words 'o', 'oh', and 'zero'.
        - 'ones': A dictionary with number names as keys and their corresponding values as integers.
        - 'ones_plural': A dictionary with number names in plural form as keys and their corresponding values and
        suffixes as tuples.
        - 'ones_ordinal': A dictionary with ordinal number names as keys and their corresponding values and
        suffixes as tuples.
        - 'ones_suffixed': A dictionary combining 'ones_plural' and 'ones_ordinal' dictionaries.
        - 'tens': A dictionary with tens names as keys and their corresponding values as integers.
        - 'tens_plural': A dictionary with tens names in plural form as keys and their corresponding values and
        suffixes as tuples.
        - 'tens_ordinal': A dictionary with tens names in ordinal form as keys and their corresponding values and
        suffixes as tuples.
        - 'tens_suffixed': A dictionary combining 'tens_plural' and 'tens_ordinal' dictionaries.
        - 'multipliers': A dictionary with multiplier names as keys and their corresponding values as integers.
        - 'multipliers_plural': A dictionary with multiplier names in plural form as keys and their corresponding values
        and suffixes as tuples.
        - 'multipliers_ordinal': A dictionary with multiplier names in ordinal form as keys and their corresponding
        values and suffixes as tuples.
        - 'multipliers_suffixed': A dictionary combining 'multipliers_plural' and 'multipliers_ordinal' dictionaries.
        - 'decimals': A set of strings representing the number names for ones, tens, and zeros.
        - 'preceding_prefixers': A dictionary with preceding prefix names as keys and their corresponding symbols
        as values.
        - 'following_prefixers': A dictionary with following prefix names as keys and their corresponding symbols
        as values.
        - 'prefixes': A set of symbols representing both preceding and following prefixes.
        - 'suffixers': A dictionary with 'per' as key and a dictionary with 'cent' as key and '%' as value.
        - 'specials': A set of special words.
        - 'words': A set of all words used for number normalization, derived from various dictionaries.
        - 'literal_words': A set of literal number words 'one' and 'ones'.
        """
        super().__init__()

        self.zeros = {"o", "oh", "zero"}
        # fmt: off
        self.ones = {
            name: i
            for i, name in enumerate(
                ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"],
                start=1,
            )
        }
        # fmt: on
        self.ones_plural = {
            "sixes" if name == "six" else name + "s": (value, "s") for name, value in self.ones.items()
        }
        self.ones_ordinal = {
            "zeroth": (0, "th"),
            "first": (1, "st"),
            "second": (2, "nd"),
            "third": (3, "rd"),
            "fifth": (5, "th"),
            "twelfth": (12, "th"),
            **{
                name + ("h" if name.endswith("t") else "th"): (value, "th")
                for name, value in self.ones.items()
                if value > 3 and value != 5 and value != 12
            },
        }
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}

        self.tens = {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        self.tens_plural = {name.replace("y", "ies"): (value, "s") for name, value in self.tens.items()}
        self.tens_ordinal = {name.replace("y", "ieth"): (value, "th") for name, value in self.tens.items()}
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}

        self.multipliers = {
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
            "sextillion": 1_000_000_000_000_000_000_000,
            "septillion": 1_000_000_000_000_000_000_000_000,
            "octillion": 1_000_000_000_000_000_000_000_000_000,
            "nonillion": 1_000_000_000_000_000_000_000_000_000_000,
            "decillion": 1_000_000_000_000_000_000_000_000_000_000_000,
        }
        self.multipliers_plural = {name + "s": (value, "s") for name, value in self.multipliers.items()}
        self.multipliers_ordinal = {name + "th": (value, "th") for name, value in self.multipliers.items()}
        self.multipliers_suffixed = {**self.multipliers_plural, **self.multipliers_ordinal}
        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "minus": "-",
            "negative": "-",
            "plus": "+",
            "positive": "+",
        }
        self.following_prefixers = {
            "pound": "£",
            "pounds": "£",
            "euro": "€",
            "euros": "€",
            "dollar": "$",
            "dollars": "$",
            "cent": "¢",
            "cents": "¢",
        }
        self.prefixes = set(list(self.preceding_prefixers.values()) + list(self.following_prefixers.values()))
        self.suffixers = {
            "per": {"cent": "%"},
            "percent": "%",
        }
        self.specials = {"and", "double", "triple", "point"}

        self.words = {
            key
            for mapping in [
                self.zeros,
                self.ones,
                self.ones_suffixed,
                self.tens,
                self.tens_suffixed,
                self.multipliers,
                self.multipliers_suffixed,
                self.preceding_prefixers,
                self.following_prefixers,
                self.suffixers,
                self.specials,
            ]
            for key in mapping
        }
        self.literal_words = {"one", "ones"}

    def process_words(self, words: List[str]) -> Iterator[str]:
        """
        Process a list of words to normalize English numbers.

        Args:
            self: Instance of the EnglishNumberNormalizer class.
            words (List[str]): A list of words representing the English number to be normalized.

        Returns:
            Iterator[str]: An iterator that yields the normalized version of the English number words.

        Raises:
            ValueError: If there is an unexpected token or if converting a fraction fails.

        Note:
            - The method normalizes English numbers by converting them into their numerical representation.
            - The normalized numbers are returned as strings.
            - The method handles various cases including prefixes, suffixes, special words, and multipliers.
            - The method supports decimal numbers.
            - The method may raise a ValueError if there is an unexpected token or if converting a fraction fails.
        """
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for i, current in enumerate(words):
            prev = words[i - 1] if i != 0 else None
            next = words[i + 1] if i != len(words) - 1 else None
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current
            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                if f is None:
                    raise ValueError("Converting the fraction failed")

                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        # concatenate decimals / ip address components
                        value = str(value) + str(current)
                        continue
                    yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator  # store integers as int
                else:
                    value = current_without_prefix
            elif current not in self.words:
                # non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = str(value or "") + "0"
            elif current in self.ones:
                ones = self.ones[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:  # replace the last zero with the digit
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
            elif current in self.ones_suffixed:
                # ordinal or cardinal; yield the number right away
                ones, suffix = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:  # eleven to nineteen
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None
            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)
            elif current in self.tens_suffixed:
                # ordinal or cardinal; yield the number right away
                tens, suffix = self.tens_suffixed[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)
            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
            elif current in self.multipliers_suffixed:
                multiplier, suffix = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:  # int
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None
            elif current in self.preceding_prefixers:
                # apply prefix (positive, minus, etc.) if it precedes a number
                if value is not None:
                    yield output(value)

                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)
            elif current in self.following_prefixers:
                # apply prefix (dollars, cents, etc.) only after a number
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)
            elif current in self.suffixers:
                # apply suffix symbols (percent -> '%')
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)
            elif current in self.specials:
                if next not in self.words and not next_is_numeric:
                    # apply special handling only if the next word can be numeric
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "and":
                    # ignore "and" after hundreds, thousands, etc.
                    if prev not in self.multipliers:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current in ('double', 'triple'):
                    if next in self.ones or next in self.zeros:
                        repeats = 2 if current == "double" else 3
                        ones = self.ones.get(next, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "point":
                    if next in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    # should all have been covered at this point
                    raise ValueError(f"Unexpected token: {current}")
            else:
                # all should have been covered at this point
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        """
        This method preprocesses the input string 's' to normalize English number representations.

        Args:
            self: The instance of the EnglishNumberNormalizer class.
            s (str): The input string containing English number representations to be preprocessed.

        Returns:
            None.

        Raises:
            None.
        """
        # replace "<number> and a half" with "<number> point five"
        results = []

        segments = re.split(r"\band\s+a\s+half\b", s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("point five")
                else:
                    results.append("and a half")

        s = " ".join(results)

        # put a space at number/letter boundary
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)

        # but remove spaces which could be a suffix
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)

        return s

    def postprocess(self, s: str):
        """
        This method postprocesses a given string to normalize English numbers and currencies.

        Args:
            self: The instance of the EnglishNumberNormalizer class.
            s (str): The input string to be postprocessed. It may contain English numbers, currencies, and
                specific patterns.

        Returns:
            str: The postprocessed string with normalized English numbers and currencies.

        Raises:
            ValueError: If the provided string contains invalid numeric values.

        The postprocess method performs the following operations on the input string:

        1. Combines the numeric values of currencies and cents into a standardized format.
        2. Extracts the cents value and replaces it with the cent symbol '¢'.
        3. Replaces the numeric representation of '1' with the word 'one'.
        """
        def combine_cents(m: Match):
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        # apply currency postprocessing; "$2 and ¢7" -> "$2.07"
        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)

        # write "one(s)" instead of "1(s)", just for the readability
        s = re.sub(r"\b1(s?)\b", r"one\1", s)

        return s

    def __call__(self, s: str):
        """
        This method normalizes English numbers in a given string.

        Args:
            self (EnglishNumberNormalizer): An instance of the EnglishNumberNormalizer class.
            s (str): The input string to be normalized.

        Returns:
            str: The normalized string with English numbers converted to their numerical form.

        Raises:
            None.
        """
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)

        return s


class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings as listed in [1].

    [1] https://www.tysto.com/uk-us-spelling-list.html
    """
    def __init__(self, english_spelling_mapping):
        """
        Initialize a new instance of the EnglishSpellingNormalizer class.

        Args:
            self (EnglishSpellingNormalizer): The instance of the class.
            english_spelling_mapping (dict): A dictionary representing the mapping of English words and their
                normalized spellings.

        Returns:
            None

        Raises:
            None
        """
        self.mapping = english_spelling_mapping

    def __call__(self, s: str):
        """
        The '__call__' method in the 'EnglishSpellingNormalizer' class normalizes the spelling of English words in a
        given string.

        Args:
            self (EnglishSpellingNormalizer): The instance of the EnglishSpellingNormalizer class.
            s (str): The input string containing English words to be normalized.

        Returns:
            None.

        Raises:
            None.
        """
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishTextNormalizer:

    """
    The EnglishTextNormalizer class represents a tool for normalizing English text by standardizing spellings, numbers,
    and formatting. It utilizes various patterns and replacement rules to clean and enhance input text.

    Attributes:
        ignore_patterns (str): Regular expression pattern to identify and ignore specific words or phrases during text
            normalization.
        replacers (dict): Dictionary mapping specific patterns to their corresponding replacements for spellings
            and contractions.
        standardize_numbers (EnglishNumberNormalizer): Instance of the EnglishNumberNormalizer class for standardizing
            numerical expressions.
        standardize_spellings (EnglishSpellingNormalizer): Instance of the EnglishSpellingNormalizer class for
            standardizing English spellings based on a provided mapping.

    Methods:
        __call__:
            Normalize the input text 's' by applying various normalization operations such as lowercase conversion,
            pattern substitution, symbol removal, and spelling standardization.

    Example:
        ```python
        >>> english_normalizer = EnglishTextNormalizer(english_spelling_mapping)
        >>> normalized_text = english_normalizer("He won't be able to make it. Let's go!")
        ```
    """
    def __init__(self, english_spelling_mapping):
        """
        Args:
            self (object): The instance of the EnglishTextNormalizer class.
            english_spelling_mapping (dict): A dictionary containing English spelling mappings.
                The keys are the original spellings, and the values are the corresponding standardized spellings.
                This parameter is used to initialize the EnglishSpellingNormalizer.
        
        Returns:
            None.
        
        Raises:
            None.
        """
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            # common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # prefect tenses, ideally it should be any past participles, but it's harder..
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",  # "'s done" is ambiguous
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer(english_spelling_mapping)

    def __call__(self, s: str):
        """
        This method '__call__' in the class 'EnglishTextNormalizer' normalizes English text by applying various
        transformations to the input string 's'.
        
        Args:
            self (EnglishTextNormalizer): An instance of the EnglishTextNormalizer class.
            s (str): The input English text to be normalized. It should be a valid string.
        
        Returns:
            str: The normalized English text after applying all the transformations.
        
        Raises:
            None
        """
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # standardize when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep some symbols for numerics

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespace characters with a space

        return s
