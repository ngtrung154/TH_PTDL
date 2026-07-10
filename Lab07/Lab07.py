import nltk
import random

nltk.download_shell()
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('names')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('tagsets')
nltk.download('tagsets_json')

print("=" * 70)
print("BAI 1: Liet ke cac ten cua corpus")
print("=" * 70)
import nltk.data
corpus_files = nltk.data.find('corpora')
import os
corpus_names = sorted([d for d in os.listdir(str(corpus_files))
                        if os.path.isdir(os.path.join(str(corpus_files), d))])
print("Cac corpus co san:", corpus_names)

print("\n" + "=" * 70)
print("BAI 2: Liet ke danh sach cac stopword bang cac ngon ngu khac nhau")
print("=" * 70)
stopwords = nltk.corpus.stopwords
print("Cac ngon ngu co san:", stopwords.fileids())
for lang in stopwords.fileids():
    print(f"\n--- {lang} ---")
    print(stopwords.words(lang)[:20])

print("\n" + "=" * 70)
print("BAI 3: Kiem tra danh sach cac stopword bang cac ngon ngu khac nhau")
print("=" * 70)
words_to_check = ['the', 'la', 'le', 'der', 'die', 'das']
for w in words_to_check:
    for lang in stopwords.fileids():
        if w in stopwords.words(lang):
            print(f"'{w}' là stopword trong ngon ngu '{lang}'")

print("\n" + "=" * 70)
print("BAI 4: Loai bo cac stopword tu mot van ban da cho")
print("=" * 70)
text = "This is a sample sentence, showing off the stop words filtration."
words = nltk.word_tokenize(text)
filtered = [w for w in words if w.lower() not in stopwords.words('english')]
print("Van ban goc:", text)
print("Sau khi loai bo stopword:", ' '.join(filtered))

print("\n" + "=" * 70)
print("BAI 5: Bo qua cac stopword tu danh sach cac stopword")
print("=" * 70)
custom_list = ['the', 'a', 'an', 'is', 'this', 'that']
filtered_stopwords = [w for w in stopwords.words('english') if w not in custom_list]
print("Stopword tieng Anh (20 dau):", stopwords.words('english')[:20])
print("Sau khi bo qua custom list (20 dau):", filtered_stopwords[:20])

print("\n" + "=" * 70)
print("BAI 6: Tim dinh nghia va vi du cua mot tu bang WordNet")
print("=" * 70)
from nltk.corpus import wordnet as wn
word = "computer"
synsets = wn.synsets(word)
print(f"Dinh nghia va vi du cua tu '{word}':")
for i, syn in enumerate(synsets[:5], 1):
    print(f"  {i}. {syn.name()} - {syn.definition()}")
    if syn.examples():
        print(f"     Vi du: {syn.examples()[0]}")

print("\n" + "=" * 70)
print("BAI 7: Tim tap hop cac tu dong nghia va trai nghia cua mot tu")
print("=" * 70)
word = "good"
synsets = wn.synsets(word)
print(f"Tu dong nghia va trai nghia cua '{word}':")
for syn in synsets[:3]:
    print(f"  Synset: {syn.name()}")
    lemmas = syn.lemmas()
    for lemma in lemmas:
        antonyms = lemma.antonyms()
        if antonyms:
            print(f"    {lemma.name()} -> trai nghia: {[a.name() for a in antonyms]}")
        else:
            print(f"    {lemma.name()}")

print("\n" + "=" * 70)
print("BAI 8: Tong quan ve bo tag, chi tiet cua mot tag cu the")
print("=" * 70)
nltk.help.upenn_tagset()
print("\n--- Chi tiet tag 'NN' ---")
nltk.help.upenn_tagset('NN')
print("\n--- Chi tiet tag 'VB' ---")
nltk.help.upenn_tagset('VB')

print("\n" + "=" * 70)
print("BAI 9: So sanh su giong nhau cua hai danh tu")
print("=" * 70)
noun1 = "car"
noun2 = "automobile"
noun3 = "dog"
syn1 = wn.synsets(noun1, pos=wn.NOUN)[0]
syn2 = wn.synsets(noun2, pos=wn.NOUN)[0]
syn3 = wn.synsets(noun3, pos=wn.NOUN)[0]
print(f"Do tuong dong giua '{noun1}' va '{noun2}': {syn1.wup_similarity(syn2):.4f}")
print(f"Do tuong dong giua '{noun1}' va '{noun3}': {syn1.wup_similarity(syn3):.4f}")

print("\n" + "=" * 70)
print("BAI 10: So sanh su giong nhau cua hai dong tu")
print("=" * 70)
verb1 = wn.synsets("run", pos=wn.VERB)[0]
verb2 = wn.synsets("sprint", pos=wn.VERB)[0]
verb3 = wn.synsets("eat", pos=wn.VERB)[0]
print(f"Do tuong dong giua 'run' va 'sprint': {verb1.wup_similarity(verb2):.4f}")
print(f"Do tuong dong giua 'run' va 'eat': {verb1.wup_similarity(verb3):.4f}")

print("\n" + "=" * 70)
print("BAI 11: Tim so luong ten nam va nu trong names corpus")
print("=" * 70)
names = nltk.corpus.names
male_names = names.words('male.txt')
female_names = names.words('female.txt')
print(f"So luong ten nam: {len(male_names)}")
print(f"So luong ten nu: {len(female_names)}")
print(f"\n10 ten nam dau tien: {male_names[:10]}")
print(f"10 ten nu dau tien: {female_names[:10]}")

print("\n" + "=" * 70)
print("BAI 12: In 15 ket hop ngau nhien duoc gan nhan nam va nu")
print("=" * 70)
labeled_names = ([(name, 'male') for name in male_names] +
                 [(name, 'female') for name in female_names])
random.shuffle(labeled_names)
print("15 ket hop ngau nhien:")
for i, (name, label) in enumerate(labeled_names[:15], 1):
    print(f"  {i}. {name} -> {label}")

print("\n" + "=" * 70)
print("BAI 13: Trich xuat ky tu cuoi cung cua tat ca ten va tao mang moi")
print("=" * 70)
labeled_features = [(name[-1], label) for name, label in labeled_names]
print(f"Tong so mau: {len(labeled_features)}")
print("20 mau dau tien (ky_tu_cuoi, nhan):")
for i, (char, label) in enumerate(labeled_features[:20], 1):
    print(f"  {i}. '{char}' -> {label}")
