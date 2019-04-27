# tokenization
Tokenization Models

## sp_uncase_en_ja_40000

A Japanese and English tokenizer with Sentencepiece model

The model were trained on texts from the cirrussearch dumps of jawiki and enwiki.

Because of memory capacity, 600,000 texts were sampled.

It contains 40,000 uncased tokens, which includes pad, unk, space and 13 unused reserved tokens.


### How this model made

See sp_uncase_en_ja_40000.ipynb jupyter notebook in sp_uncase_en_ja_40000 directory.


### Usage

Make sure that sentencepiece library is installed before using this script.

```bash
pip install sentencepiece
```

```python
from tokenization.sp_uncase_en_ja_40000 import FullTokenizer
tokenizer = FullTokenizer()

text1 = "信じられているから走るのだ。間に合う、間に合わぬは問題でないのだ。"
text2 = "新たな時代のMarxよこれらの盲目な衝動から動く世界を素晴しく美しい構成に変へよ"
text3 = 'The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.'

tokenizer(text1)
# ['信', 'じ', 'られている', 'から', '走る', 'のだ', '。', '間に', '合う', '、', '間に', '合', 'わ', 'ぬ', 'は', '問題', 'でない', 'のだ', '。']
tokenizer(text1, as_ids=True)
# [814, 1165, 3284, 102, 20365, 20187, 21, 10373, 13545, 20, 10373, 1500, 1639, 6513, 36, 2007, 15117, 20187, 21]

tokenizer(text2)
# ['新たな', '時代の', 'marx', 'よ', 'これらの', '盲', '目', 'な', '衝', '動', 'から', '動く', '世界', 'を', '素', '晴', 'しく', '美しい', '構成', 'に', '変', 'へ', 'よ']
tokenizer(text2, as_ids=True)
# [8137, 3376, 26253, 1582, 5865, 22135, 863, 228, 20725, 2749, 102, 30524, 681, 46, 3243, 4225, 6853, 11727, 3595, 45, 3949, 381, 1582]

tokenizer(text3)
# ['the', '?', 'above', '?', 'copyright', '?', 'notice', '?', 'and', '?', 'this', '?', 'permission', '?', 'notice', '?', 'shall', '?', 'b', 'e', '?', 'included', '?', 'in', '?', 'all', '?', 'copies', '?', 'or', '?', 'substantial', '?', 'portion', 's', '?', 'of', '?', 'the', '?', 'software', '.']
# ? is (U+2581), which represents whitespace in sentencepiece
tokenizer(text3, remove_spaces=True)
# ['the', 'above', 'copyright', 'notice', 'and', 'this', 'permission', 'notice', 'shall', 'b', 'e', 'included', 'in', 'all', 'copies', 'or', 'substantial', 'portion', 's', 'of', 'the', 'software', '.']
```

