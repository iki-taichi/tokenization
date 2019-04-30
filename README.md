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


## sp_uncase_en_30000

[Notebook for training](https://github.com/iki-taichi/tokenization/blob/master/sp_uncase_en_30000/sp_uncase_en_30000.ipynb)

```python
>>> text1='Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, `and what is the use of a book,’ thought Alice `without pictures or conversation?’'
>>>
>>> text2='So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.'
>>>
>>> text3='There was nothing so VERY remarkable in that; nor did Alice think it so VERY much out of the way to hear the Rabbit say to itself, `Oh dear! Oh dear! I shall be late!’ (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite naural); but when the Rabbit actually TOOK A WATCH OUT OF ITS WAISTCOAT- POCKET, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.'
>>>
>>> from tokenization.sp_uncase_en_30000 import FullTokenizer
>>> tokenizer = FullTokenizer()
>>> tokenizer(text1)
['▁alice', '▁was', '▁beginning', '▁to', '▁get', '▁very', '▁tir', 'ed', '▁of', '▁sitting', '▁by', '▁her', '▁sister', '▁on', '▁the', '▁bank', ',', '▁and', '▁of', '▁having', '▁nothing', '▁to', '▁do', ':', '▁once', '▁or', '▁twice', '▁she', '▁had', '▁pee', 'ped', '▁into', '▁the', '▁book', '▁her', '▁sister', '▁was', '▁reading', ',', '▁but', '▁it', '▁had', '▁no', '▁pictures', '▁or', '▁conversation', 's', '▁in', '▁it', ',', '▁', '`', 'and', '▁what', '▁is', '▁the', '▁use', '▁of', '▁a', '▁book', ',', '’', '▁thought', '▁alice', '▁', '`', 'without', '▁pictures', '▁or', '▁conversation', '?', '’']
>>> tokenizer(text2)
['▁so', '▁she', '▁was', '▁considering', '▁in', '▁her', '▁own', '▁mind', '▁(', 'as', '▁well', '▁as', '▁she', '▁could', ',', '▁for', '▁the', '▁hot', '▁day', '▁made', '▁her', '▁feel', '▁very', '▁sleep', 'y', '▁and', '▁stupid', '),', '▁whether', '▁the', '▁pleasure', '▁of', '▁making', '▁a', '▁daisy', '-', 'chain', '▁would', '▁be', '▁worth', '▁the', '▁trouble', '▁of', '▁getting', '▁up', '▁and', '▁pick', 'ing', '▁the', '▁dai', 's', 'ies', ',', '▁when', '▁suddenly', '▁a', '▁white', '▁rabbit', '▁with', '▁pink', '▁eyes', '▁ran', '▁close', '▁by', '▁her', '.']
>>> tokenizer(text3)
['▁there', '▁was', '▁nothing', '▁so', '▁very', '▁remarkable', '▁in', '▁that', ';', '▁nor', '▁did', '▁alice', '▁think', '▁it', '▁so', '▁very', '▁much', '▁out', '▁of', '▁the', '▁way', '▁to', '▁hear', '▁the', '▁rabbit', '▁say', '▁to', '▁itself', ',', '▁', '`', 'oh', '▁dear', '!', '▁oh', '▁dear', '!', '▁i', '▁shall', '▁be', '▁late', '!', '’', '▁(', 'when', '▁she', '▁thought', '▁it', '▁over', '▁afterwards', ',', '▁it', '▁occurred', '▁to', '▁her', '▁that', '▁she', '▁o', 'ught', '▁to', '▁have', '▁wonder', 'ed', '▁at', '▁this', ',', '▁but', '▁at', '▁the', '▁time', '▁it', '▁all', '▁seemed', '▁quite', '▁natural', ');', '▁but', '▁when', '▁the', '▁rabbit', '▁actually', '▁took', '▁a', '▁watch', '▁out', '▁of', '▁its', '▁wa', 'ist', 'coat', '-', '▁pocket', ',', '▁and', '▁looked', '▁at', '▁it', ',', '▁and', '▁then', '▁hur', 'ried', '▁on', ',', '▁alice', '▁started', '▁to', '▁her', '▁feet', ',', '▁for', '▁it', '▁flash', 'ed', '▁across', '▁her', '▁mind', '▁that', '▁she', '▁had', '▁never', '▁before', '▁seen', '▁a', '▁rabbit', '▁with', '▁either', '▁a', '▁wa', 'ist', 'coat', '-', 'p', 'ock', 'et', ',', '▁or', '▁a', '▁watch', '▁to', '▁take', '▁out', '▁of', '▁it', ',', '▁and', '▁burning', '▁with', '▁cur', 'ios', 'ity', ',', '▁she', '▁ran', '▁across', '▁the', '▁field', '▁after', '▁it', ',', '▁and', '▁fortuna', 'te', 'ly', '▁was', '▁just', '▁in', '▁time', '▁to', '▁see', '▁it', '▁pop', '▁down', '▁a', '▁large', '▁rabbit', '-', 'hole', '▁under', '▁the', '▁hedge', '.']
```

## sp_uncase_ja_30000

[Notebook for training](https://github.com/iki-taichi/tokenization/blob/master/sp_uncase_ja_30000/sp_uncase_ja_30000.ipynb)

```python
>>> text1='　私はその人を常に先生と呼んでいた。だからここでもただ先生と書くだけで本名は打ち明けない。これは世間を憚かる遠慮というよりも、その方が私にとって自 然だからである。私はその人の記憶を呼び起すごとに、すぐ「先生」といいたくなる。筆を執っても心持は同じ事である。よそよそしい頭文字などはとても使う気にならない。'
>>>
>>> text2='　私が先生と知り合いになったのは鎌倉である。その時私はまだ若々しい書生であった。暑中休暇を利用して海水浴に行った友達からぜひ来いという端書を受け取 ったので、私は多少の金を工面して、出掛ける事にした。私は金の工面に二、三日を費やした。ところが私が鎌倉に着いて三日と経たないうちに、私を呼び寄せた友達は、急に国元から帰れという電報を受け取った。電報には母が病気だからと断ってあったけれども友達はそれを信じなかった。友達はかねてから国元にいる親たちに勧まない結婚を強いられていた。彼は現代の習慣からいうと結婚するにはあまり年が若過ぎた。それに肝心の当人が気に入らなかった。それで夏休みに当然帰るべきところを、わざと避けて東京の近くで遊んでいたのである。彼は電報を私に見せてどうしようと相談をした。私にはどうしていいか分らなかった。けれども実際彼の母が病気であるとすれば彼は固より帰るべきはずであった。それで彼はとうとう帰る事になった。せっかく来た私は一人取り残された。'
>>>
>>> from tokenization.sp_uncase_ja_30000 import FullTokenizer
>>> tokenizer = FullTokenizer()
>>> tokenizer(text1)
['私は', 'その', '人を', '常に', '先生', 'と呼んで', 'いた', '。', 'だから', 'ここで', 'も', 'ただ', '先生', 'と', '書く', 'だけで', '本名は', '打ち', '明', 'けない', '。', 'これは', '世間', 'を', '憚', 'かる', '遠', '慮', 'という', 'よりも', '、', 'その', '方が', '私', 'にとって', '自然', 'だから', 'である', '。', '私は', 'その', '人', 'の', '記憶', 'を呼び', '起', 'す', 'ごとに', '、', 'すぐ', '「', '先生', '」', 'といい', 'た', 'くなる', '。', '筆', 'を', '執', 'っても', '心', '持', 'は同じ', '事', 'である', '。', 'よ', 'そ', 'よ', 'そ', 'しい', '頭', '文字', 'などは', 'とても', '使う', '気', 'に', 'ならない', '。']
>>> tokenizer(text2)
['私', 'が', '先生', 'と', '知り合い', 'になった', 'の', 'は', '鎌倉', 'である', '。', 'その時', '私', 'はまだ', '若', '々', 'しい', '書', '生', 'であった', '。', '暑', '中', '休暇', 'を利用して', '海水', '浴', 'に行った', '友達', 'から', 'ぜ', 'ひ', '来', 'い', 'という', '端', '書', 'を受け', '取った', 'の', 'で', '、', '私は', '多少', 'の', '金', 'を', '工', '面', 'して', '、', '出', '掛け', 'る事', 'にした', '。', '私は', '金', 'の', '工', '面', 'に', '二', '、', '三', '日', 'を', '費', 'や', 'した', '。', 'ところが', '私', 'が', '鎌倉', 'に', '着', 'いて', '三', '日', 'と', '経', 'た', 'ない', 'うちに', '、', '私', 'を呼び', '寄せ', 'た', '友達', 'は', '、', '急', 'に', '国', '元', 'から', '帰', 'れ', 'という', '電報', 'を受け', '取った', '。', '電報', 'には', '母', 'が', '病気', 'だから', 'と', '断', 'って', 'あった', 'け', 'れ', 'ど', 'も', '友達', 'は', 'それを', '信', 'じ', 'なかった', '。', '友達', 'は', 'かね', 'て', 'から', '国', '元', 'にいる', '親', 'たち', 'に', '勧', 'ま', 'ない', '結婚', 'を強いられ', 'ていた', '。', '彼は', '現代', 'の', '習慣', 'から', 'いう', 'と結婚', 'する', 'に', 'はあまり', '年', 'が', '若', '過ぎ', 'た', '。', 'それに', '肝', '心', 'の', '当', '人が', '気', 'に入', 'らなかった', '。', 'それ', 'で', '夏休み', 'に', '当然', '帰', 'るべき', 'ところ', 'を', '、', 'わざ', 'と', '避け', 'て', '東京', 'の', '近く', 'で', '遊', 'んでいた', 'の', 'である', '。', '彼は', '電報', 'を', '私', 'に', '見せ', 'て', 'どう', 'しようと', '相談', 'をした', '。', '私', 'には', 'どう', 'して', 'いい', 'か', '分', 'らなかった', '。', 'け', 'れ', 'ど', 'も', '実際', '彼の', '母', 'が', '病気', 'である', 'と', 'すれば', '彼は', '固', 'より', '帰', 'るべき', 'はず', 'であった', '。', 'それ', 'で', '彼は', 'とう', 'とう', '帰る', '事', 'になった', '。', 'せ', 'っか', 'く', '来た', '私は', '一人', '取り', '残された', '。']
```

