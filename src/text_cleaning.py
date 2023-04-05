import re
from num2words import num2words
from unicode_tr import unicode_tr
import string
from tqdm import tqdm
tqdm.pandas()

mapping_toxic_words = {
    'ag': 'amına koyayım', 'got': 'göt', 'gotten': 'götten', 'gotune': 'götüne',
    'gote': 'göte', 'gotunde': 'götünde', 'gotunden': 'götünden', 'gotunu': 'götünü', 'gotu': 'götü', 'gotun': 'götün',
    'ami': 'amı', 'aminda': 'amında', 'amina': 'amına', 'amini': 'amını', 'yarragm': 'yarak', 'yrrk': 'yarak',
    'skrm': 'sikerim', 'skrim': 'sikerim', 's1kerim': 'sikerim', 's1ker1m': 'sikerim', 'gtn': 'göt', 'daşşak': 'taşak',
    'yaraaam': 'yarak', 'totoş': 'göt', 'amina': 'amına', 'zik': 'yarak', 'zikiim': 'yarak', 'yarragm': 'yarak',
    'sikiir': 'yarak',
    'amuğa': 'amına', 'amk': 'amına koyayım', "a.q": "amına koyayım", "a.w": "amına koyayım", "ak": "amına koyayım",
    "a.k": "amına koyayım",
    "mq": "amına koyayım", "mk": "amına koyayım", "aqqq": "amına koyayım", "awk": "amına koyayım",
    'oc': 'orospu çocuğu', 'ass': 'göt',
    'OÇ': 'orospu çocuğu', 'pic': 'piç', 'a.q': 'amına koyayım', 'aq.': 'amına koyayım', 'kappe': 'kahpe',
    'amnn': 'amına', 'amn': 'amına',
    'pussy': 'amcık', 'sikt': 'siktir', 'sksz': 'seks', 'amck': 'amcık', 'ferre': 'seks', 'sexs': 'seks',
    'yrrak': 'yarak', 'mcik': 'amcık', 'sikm': 'sikeyim', 'amna': 'amına',
    'pezo': 'pezevenk', 'saxo': 'seks', 'götü': 'göt', 'skem': 'sikeyim', 'yarraaaa': 'yarak', 'sktrr': 'siktir',
    'sike': 'yarak', 'skiim': 'sikeyim', 'skim': 'sikeyim',
    'sikik': 'yarak', 'yarrağ': 'yarak', "amk": "amına koyayım", "aq": "amına koyayım", 'ık': 'amına koyayım',
    "fuck": "siktir", "ananskm": "ananı sikim",
    "ziktir": "siktir", "amkdjdkd": "amına koyayım", "yardagıma": "yarrağıma", 'çük': 'yarak', 'penis': 'yarak',
    'pipi': 'yarak', 'sik': 'yarak', 'yarrak': 'yarak',
    'kaka': 'bok', 'oç': 'oruspu çocuğu', 'wtf': 'siktir', 'sie': 'siktir git', 'kürd': 'kürt', 'kurdo': 'kürt',
    'kürdo': 'kürt', 'turko': 'türk',
    'türko': 'türk', 'birşey': 'bir şey', 'herşey': 'her şey', 'yanlız': 'yalnız', 'yalnış': 'yanlış',
    'herkez': 'herkes', 'bir kaç': 'birkaç',
    'birebir': 'bire bir', 'birşeyler': 'bir şeyler', 'yavsak': 'yavşak', 'oruspu': 'orospu', 'orspu': 'orospu',
    'bişey': 'bir şey', 'hiçbişey': 'hiçbir şey',
    'motherfucker': 'orospu çocuğu', 'fucker': 'siken', 'shit': 'siktir', 'fuckin': 'siktir',
    'fucking': 'siktir', 'fuck': 'siktir', 'ass': 'göt',
    'bitch': 'orospu', 'idiot': 'aptal', 'pussy': 'amcık', 'sikiyim': 'sikeyim', 'got': 'göt', 'kahbe': 'kahpe',
    'gurub': 'grup',
    'ıbne': 'ibne', 'pust': 'puşt', 'ibn': 'ibne', 'mcık': 'amcık', 'mcik': 'amcık', 's.k': 'sik kafalı', 'g.t': 'göt',
    's.ktir': 'siktir', 'yawşak': 'yavşak', 'yawsak': 'yavşak', 'pic': 'piç', '31': 'mastürbasyon', 'kerane': 'kerhane', 'koyim': 'koyayım',
    'g.tü': 'götü', 'y.rrak': 'yarak', '*mcık': 'amcık', '*mcik': 'amcık', 'mına': 'amına', 'göd': 'göt', 'ancık': 'amcık',
    'xik': 'yarak', 'ipne': 'ibne', 's*x': 'seks', 'orsbu': 'orospu', 'orrspu': 'orospu', 'mcını': 'amını', 'zittin': 'siktin',
    'zittim': 'siktim', 'inbe': 'ibne', 'amuğa': 'amına', 'yarra': 'yarak', 'mını': 'amını', 'yarraa': 'yarak', 'a.q.': 'amına koyayım',
    'a.k.': 'amına koyayım', 'amnskm': 'amını sikeyim',
    'amık': 'amcık', 'ybsg': 'siktir git', 'bsg': 'siktir git', "amınako": "amına koyarım", "sikem": 'sikerim',
    "zigsin": 'siksin', "zikeyim": 'sikeyim', "zikiiim": 'yarak', "zikiim": 'yarak', "zikik": 'yarak', "zikim": 'yarak',
    "ziksiiin": 'siksin', "ziksiin": 'siksin',
    "yaram": 'yarak', "yaraminbasi": 'yarak', "yaramn": 'yarak', "yarraak": 'yarak', "yarraam": 'yarak',
    "yarraamı": 'yarak', "yarragi": 'yarak',
    "sittimin": 'siktiğimin', "sittir": 'siktir', "skcem": 'sikeceğim', "skecem": 'sikeceğim', "skem": 'sikerim',
    "sker": 'siker', "skerim": 'sikerim', "skerm": 'sikerim',
    "skeyim": 'sikeyim', "skiim": 'sikeyim', "skik": 'yarak', "skim": 'yarak', "skime": 'yarak', "skmek": 'sikmek',
    "sksin": 'siksin', "sksn": 'siksin', "sksz": 'sikerim', "sktiimin": 'siktiğimin', "sktrr": 'siktir',
    "skyim": 'sikeyim', "sokam": 'sokarım', "yarragimi": 'yarak', "yarragina": 'yarak', "yarragindan": 'yarak', "yarragm": 'yarak',
    "yarrağ": 'yarak', "yarrağım": 'yarak', "yarrağımı": 'yarak', "yarraimin": 'yarak',
    'sikim': 'sikeyim', 'sex': 'seks', 'daşak': 'taşak', 'taşşak': 'taşak', 'amcik': 'amcık', 'oruspiy': 'orospu',
    'amina': 'amına', 'amindan': 'amından', 'escort': 'eskort',
    'am': 'amcık', 'koyim': 'koyayım', 'sg': 'siktir git', 'siktirgit': 'siktir git', 'serefsiz': 'şerefsiz'
}


def toxic_mapping(text):
    toxic_map_text_list = []
    for word in text.split():
        if word in mapping_toxic_words:
            toxic_map_text_list.append(mapping_toxic_words[word])
        else:
            toxic_map_text_list.append(word)
    toxic_map_text_list = " ".join(toxic_map_text_list)
    return toxic_map_text_list


def remove_quest_suf_sep(text):
    s = []
    for txt in text.split():
        if re.search('(mısın|misin|musun|müsün)$', txt):
            suffix = re.search('(mısın|misin|musun|müsün)$', txt).group()
            s.append(re.sub('(mısın|misin|musun|müsün)$', '', txt) + ' ' + suffix)
        else:
            s.append(txt)
    return ' '.join(s)


def remove_single_chars(text):
    return ' '.join([w for w in text.split() if len(w) > 1 or w in ['o', 'O']])


def remove_tweets(text):
	# remove mentions and tags
	text = re.sub("@[A-Za-z0-9_]+", "", text)
	text = ' '.join(word for word in text.split() if not word[0] == "#")
	# remove url
	text = re.sub(r"http\S+", "", text)
	text = re.sub('http[s]?://\S+', '', text)
	text = re.sub('http://\S+|https://\S+', '', text)    
	text = re.sub(r'<[^>]+>', '', text)
	text = re.sub(r'http\S+', '', text)
	return text

def clean_text(df):
    
    df_copy = df.copy() 
    # lower text
    df_copy['text'] = df_copy.text.progress_apply(lambda x: str(x))
    df_copy['lower_text'] = df_copy.text.apply(lambda x: unicode_tr(x).lower())
    
    # fix and map molds
    df_copy['toxic_mapping'] = df_copy.lower_text.progress_apply(toxic_mapping)
    
    # clean question suffixes
    df_copy['remove_quest_suf_sep'] = df_copy.toxic_mapping.progress_apply(remove_quest_suf_sep)
    
    # clean tweet signatures as @, # etc.
    df_copy['clean_tweet_sign'] = df_copy['remove_quest_suf_sep'].progress_apply(remove_tweets)
    
    # remove multiple spaces
    df_copy['remove_multiple_spaces'] = df_copy['clean_tweet_sign'].progress_apply(lambda x: re.sub(' +', ' ', x))
    
    # number to text
    df_copy['number_to_text'] = df_copy['remove_multiple_spaces'].progress_apply(lambda x: ' '.join([num2words(i, lang='tr') if i.isnumeric() else i for i in x.split()]))
    
    # remove daki, deki suffixes
    df_copy['remove_deki'] = df_copy.number_to_text.progress_apply(lambda x: ' '.join([re.sub(r'd[ae]ki', '', i) if i.endswith(('deki', 'daki')) else i for i in x.split()]))
    
    # remove punctuations
    df_copy['remove_punc'] = df_copy.remove_deki.progress_apply(lambda x: ''.join(filter(lambda c: c not in string.punctuation, x)))
    
    df_copy = df_copy.drop(['lower_text', 'toxic_mapping', 'remove_quest_suf_sep', 'clean_tweet_sign', 'remove_multiple_spaces'], axis=1)
    
    return df_copy