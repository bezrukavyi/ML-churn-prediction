import pandas as pd
import numpy as np
from steps.load_data import LoadData
import pickle


task = 0
task_total = 0


# TRAIN bnum
def generate_train_bnum_feature(dataframe):
    cache_key = "cache/bnum_features(short)_train_2.pkl"

    bnum_new_df = generate_bnum_features(LoadData().df_train_fe, LoadData().df_train_bnum)

    with open(cache_key, "wb") as f:
        pickle.dump(bnum_new_df, f)

    with open(cache_key, "rb") as f:
        bnum_new_df = pickle.load(f)

    return dataframe.merge(bnum_new_df, on="abon_id", how="left").fillna(0)


# TEST bnum
def generate_test_bnum_feature(dataframe):
    cache_key = "cache/bnum_features(short)_test_2.pkl"

    bnum_new_df = generate_bnum_features(LoadData().df_test_fe, LoadData().df_test_bnum)

    with open(cache_key, "wb") as f:
        pickle.dump(bnum_new_df, f)

    with open(cache_key, "rb") as f:
        bnum_new_df = pickle.load(f)

    return dataframe.merge(bnum_new_df, on="abon_id", how="left").fillna(0)


def generate_bnum_features(dataframe, bnum_df):
    global task
    global task_total

    # Load data
    bnum_df = bnum_df.copy()

    task = 0
    task_total = len(bnum_df.abon_id.unique())

    # Assign target
    bnum_df["target"] = 0
    abon_target = dataframe[["target", "abon_id"]].copy()
    bnum_df["target"] = bnum_df["abon_id"].map(abon_target.set_index("abon_id")["target"])
    bnum_df["abon_id"] = bnum_df["abon_id"].astype(int)

    # Assign bnum category
    bnum_df = assign_bnum_category(bnum_df, bnum_categories_dict)

    # Split the DataFrame into groups by abon_id
    df_new = bnum_df.groupby("abon_id").apply(build_user_bnum_metrics_group).reset_index(drop=True)

    # Format abon_id
    df_new["abon_id"] = df_new["abon_id"].astype(int)

    return df_new


def calculate_share(value, total):
    return (value / total) * 100 if total > 0 else 0


def call_cnt_out_metrics(group, agg_data):
    bnum_topic_agg_sum = agg_data["bnum_topic_agg_sum"]["call_cnt_out"]
    bnum_category_agg_sum = agg_data["bnum_category_agg_sum"]["call_cnt_out"]

    bnum_topic_sum = bnum_topic_agg_sum.sum()

    metrics = {}

    metrics["vodafone_topic_call_cnt_out_sum"] = bnum_topic_agg_sum.get("vodafone_topic", 0)
    metrics["vodafone_topic_call_cnt_out_sum_share"] = calculate_share(
        metrics["vodafone_topic_call_cnt_out_sum"], bnum_topic_sum
    )
    metrics["finance_topic_call_cnt_out_sum"] = bnum_topic_agg_sum.get("finance_topic", 0)
    metrics["rescue_numbers_call_cnt_out_sum"] = bnum_category_agg_sum.get("rescue_numbers", 0)
    metrics["bank_numbers_call_cnt_out_sum"] = bnum_category_agg_sum.get("bank_numbers", 0)
    metrics["fraud_numbers_call_cnt_out_sum"] = bnum_category_agg_sum.get("fraud_numbers", 0)
    metrics["casual_topic_call_cnt_out_sum"] = bnum_topic_agg_sum.get("casual_topic", 0)

    metrics["vodafone_act_call_cnt_out"] = agg_data["vodafone_df"]["call_cnt_out"].sum()
    metrics["casual_act_call_cnt_out"] = agg_data["casual_df"]["call_cnt_out"].sum()

    return metrics


def call_cnt_in_metrics(group, agg_data):
    bnum_topic_agg_sum = agg_data["bnum_topic_agg_sum"]["call_cnt_in"]

    metrics = {}

    metrics["casual_topic_call_cnt_in_sum"] = bnum_topic_agg_sum.get("casual_topic", 0)

    return metrics


def call_dur_out_metrics(group, agg_data):
    bnum_topic_agg_sum = agg_data["bnum_topic_agg_sum"]["call_dur_out"]
    bnum_category_agg_sum = agg_data["bnum_category_agg_sum"]["call_dur_out"]

    metrics = {}

    metrics["vodafone_topic_call_dur_out_sum"] = bnum_topic_agg_sum.get("vodafone_topic", 0)
    metrics["finance_topic_call_dur_out_sum"] = bnum_topic_agg_sum.get("finance_topic", 0)
    metrics["casual_topic_call_dur_out_sum"] = bnum_topic_agg_sum.get("casual_topic", 0)
    metrics["competitors_topic_call_dur_out_sum"] = bnum_topic_agg_sum.get("competitors_topic", 0)
    metrics["fraud_numbers_call_dur_out_sum"] = bnum_category_agg_sum.get("fraud_numbers", 0)

    metrics["vodafone_act_call_dur_out"] = agg_data["vodafone_df"]["call_dur_out"].sum()
    metrics["casual_act_call_dur_out"] = agg_data["casual_df"]["call_dur_out"].sum()

    return metrics


def cnt_sms_out_metrics(group, agg_data):
    bnum_topic_agg_sum = agg_data["bnum_topic_agg_sum"]["cnt_sms_out"]
    bnum_category_agg_sum = agg_data["bnum_category_agg_sum"]["cnt_sms_out"]

    bnum_category_sum = bnum_category_agg_sum.sum()
    bnum_topic_sum = bnum_topic_agg_sum.sum()

    metrics = {}

    metrics["casual_topic_cnt_sms_out_sum"] = bnum_topic_agg_sum.get("casual_topic", 0)
    metrics["casual_topic_cnt_sms_out_sum_share"] = calculate_share(
        metrics["casual_topic_cnt_sms_out_sum"], bnum_topic_sum
    )

    metrics["other_cnt_sms_out_sum"] = bnum_category_agg_sum.get("other", 0)
    metrics["other_cnt_sms_out_sum_share"] = calculate_share(metrics["other_cnt_sms_out_sum"], bnum_category_sum)

    metrics["vodafone_act_cnt_sms_out"] = agg_data["vodafone_df"]["cnt_sms_out"].sum()
    metrics["casual_act_cnt_sms_out"] = agg_data["casual_df"]["cnt_sms_out"].sum()

    return metrics


def cnt_sms_in_metrics(group, agg_data):
    bnum_topic_agg_sum = agg_data["bnum_topic_agg_sum"]["cnt_sms_in"]
    bnum_category_agg_sum = agg_data["bnum_category_agg_sum"]["cnt_sms_in"]

    bnum_category_sum = bnum_category_agg_sum.sum()
    bnum_topic_sum = bnum_topic_agg_sum.sum()

    metrics = {}

    metrics["other_cnt_sms_in_sum"] = bnum_category_agg_sum.get("other", 0)
    metrics["bank_companies_cnt_sms_in_sum"] = bnum_category_agg_sum.get("bank_companies", 0)
    metrics["vodafone_services_cnt_sms_in_sum"] = bnum_category_agg_sum.get("vodafone_services", 0)
    metrics["vodafone_survey_cnt_sms_in_sum"] = bnum_category_agg_sum.get("vodafone_survey", 0)

    metrics["other_cnt_sms_in_sum_share"] = calculate_share(metrics["other_cnt_sms_in_sum"], bnum_category_sum)
    metrics["vodafone_services_cnt_sms_in_sum_share"] = calculate_share(
        metrics["vodafone_services_cnt_sms_in_sum"], bnum_category_sum
    )
    metrics["bank_companies_cnt_sms_in_sum_share"] = calculate_share(
        metrics["bank_companies_cnt_sms_in_sum"], bnum_category_sum
    )

    metrics["casual_topic_cnt_sms_in_sum"] = bnum_topic_agg_sum.get("casual_topic", 0)
    metrics["finance_topic_cnt_sms_in_sum"] = bnum_topic_agg_sum.get("finance_topic", 0)

    metrics["casual_topic_cnt_sms_in_sum_share"] = calculate_share(
        metrics["casual_topic_cnt_sms_in_sum"], bnum_topic_sum
    )
    metrics["vodafone_services_cnt_sms_in_sum_share"] = calculate_share(
        metrics["vodafone_services_cnt_sms_in_sum"], bnum_category_sum
    )
    metrics["finance_topic_cnt_sms_in_sum_share"] = calculate_share(
        metrics["finance_topic_cnt_sms_in_sum"], bnum_topic_sum
    )

    metrics["vodafone_act_cnt_sms_in"] = agg_data["vodafone_df"]["cnt_sms_in"].sum()
    metrics["casual_act_cnt_sms_in"] = agg_data["casual_df"]["cnt_sms_in"].sum()

    return metrics


def build_user_bnum_metrics_group(group):
    global task
    global task_total

    bnum_topic_agg_sum = group.groupby("bnum_topic").sum()
    bnum_topic_agg_count = group.groupby("bnum_topic").count()
    bnum_category_agg_sum = group.groupby("bnum_category").sum()
    bnum_category_agg_count = group.groupby("bnum_category").count()

    agg_data = {
        "bnum_topic_agg_sum": bnum_topic_agg_sum,
        "bnum_category_agg_sum": bnum_category_agg_sum,
        "vodafone_df": group[group.vodafone == True],
        "casual_df": group[group.casual == True],
    }

    metrics = {
        "abon_id": group.iloc[0].abon_id,
        **call_cnt_out_metrics(group, agg_data),
        **call_cnt_in_metrics(group, agg_data),
        **call_dur_out_metrics(group, agg_data),
        **cnt_sms_out_metrics(group, agg_data),
        **cnt_sms_in_metrics(group, agg_data),
    }

    # print(f"Task {task}/{task_total} is done")
    task += 1
    print(f"Progress: {task/task_total*100:.2f}%")

    return pd.Series(metrics)


bank_companies = [
    "privatbank",
    "oschadbank",
    "raiffeisen",
    "monobank",
    "mono",
    "alfabank",
    "pumb",
    "otp bank",
    "ukrgasbank",
    "a-bank",
    "tascombank",
    "ukreximbank",
    "bankvostok",
    "pivdenny",
    "globusbank",
    "ukreximban",
    "poltavaban",
    "bank bls",
    "accordbank",
    "creditdnepr",
    "megabank",
    "concordbank",
    "radabank",
    "procredit",
    "industrial",
    "pravex ban",
    "rws bank",
    "bankago",
    "credobank",
    "forwardban",
    "sportbank",
    "idea bank",
    "mtb bank",
    "piraeusbank",
    "bankforward",
    "alf-ua.com",
    "kredobank",
    "sichbank",
    "izibank",
    "pumb onlin",
    "ideabank",
    "cagricole",
]

work_companies = [
    "rabota.ua",
    "work.ua",
    "jooble",
    "linkedin",
    "itstep",
    "skillup",
    "rabota",
    "start-work",
]

grocery_companies = [
    "varus",
    "silpo",
    "fozzy",
    "myasomarke",
    "metro",
    "tavria_v",
    "eko market",
    "фрукты",
]

post_companies = [
    "380984500609",
    "novaposhta",
]

taxy_companies = [
    "taxi-838",
    "uklon",
    "bolt",
    "uber",
    "shark taxi",
    "taxi 3040",
    "opti-579",
    "ontaxi",
    "taxi 2288",
    "taxi vezi",
    "taxi 323",
    "taxi 571",
    "taxi-808",
    "eco taxi",
    "taxi-280",
    "taximer",
    "taxi 309",
    "taxi 777",
    "taxi-653",
    "taxi 959",
    "2233",
    "3135",
    "3133",
]

credits_companies = [
    "credit_plus",
    "credit7.ua",
    "mycredit",
    "ze.kredit",
    "moneyveo",
    "mycredit.ua",
    "creditdnepr",
    "e-groshi",
    "credit7",
    "clickcredit",
    "soscredit",
    "crediton",
    "creditexpr",
    "creditdnep",
    "credit_plus",
    "creditplu",
    "creditdebt",
    "zecredit",
    "kredit",
    "alexcredit",
    "shgroshi",
    "creditpod",
    "credit7.ua",
    "money4you",
    "dinero",
    "kreditytut",
    "creditpod-0",
    "clickcredi",
    "kredit4u",
    "e-wings",
    "creditexpr",
    "minizaem",
    "money24",
    "moneyboom",
    "moneyextra",
    "moneylove",
    "zss",
    "creditplus",
    "k-kapital",
    "egroshicom",
    "skarbcomua",
    "credit_plu",
    "miloan.ua",
    "mycredit-u",
    "mycredit-ua",
    "mycredit.u",
    "ua-mycredi",
    "ua-mycredit",
    "mycreditua",
    "ccloan",
    "loanyua",
    "ze.kred1t",
    "selficredi",
    "mistercash",
    "selficredit",
    "sos credit",
    "zecredi",
    "dengi dozp",
    "otp credit",
    "ze_kred1t",
    "sloncredit",
    "bingocash",
    "cash point",
    "cashberry",
    "slon credi",
    "domonet",
    "yoomoney",
    "creditpod-",
    "slon credit",
    "moneta-z",
    "bankacredi",
    "cash-kf",
    "advcash",
    "money.4.yo",
    "moneyexpert",
    "money-4-yo",
    "aviracredi",
    "money-4-you",
    "icredit",
    "creditbox",
    "money.4.you",
    "credit",
    "zekredi",
    "el caso",
    "creditup",
    "monetka",
    "dengivsim",
    "credit365",
    "moneyexper",
    "aviracredit",
    "kredit plu",
    "crediglbl",
    "bestcredit",
    "neocredit",
    "cashhelp",
    "telmone",
    "microcash",
    "novikredyt",
    "extramoney",
    "clycredit",
    "creditcafe",
    "ewacash",
    "kg-money",
    "creditik",
    "casharing",
    "case 24",
    "kreditstar",
    "forzacredit",
    "webmoney.ua",
    "forzacredi",
    "kredit plus",
    "ecase",
    "caseshop",
    "webmoney.u",
    "macincase",
    "verocash",
    "techno-cas",
    "novikredyty",
    "cash24",
    "onlycredit",
    "portmone",
    "mrmoney",
    "globalcredi",
    "credit-pro",
    "cashdesk",
    "stormoney",
    "credit2u",
    "opencredit",
    "smartmoney",
    "kvk-cash",
    "hotcredit",
    "techno-case",
    "cash365",
    "bestcredits",
    "bankacredit",
    "cash-ua",
    "sweet mone",
    "kredit1",
    "cashinua",
    "sweetmoney",
    "tviy cash",
    "moneyua",
    "catcredit",
    "money.spac",
    "topcredit",
    "sweet money",
    "pvks-kredit",
    "fotocaseua",
    "cashinsky",
    "easy cash",
    "yourmoney",
    "credbox",
    "kreditavans",
    "credit24",
    "casinoin",
    "globalcred",
    "caseller",
    "n1casino",
    "recredit",
    "happycredi",
    "uitracredit",
    "onecase",
    "seacredit",
    "smartcredi",
    "moneysmash",
    "pvks-kredi",
    "gdcashmere",
    "cool credi",
    "mlcrocredit",
    "turbocash",
    "allrightcas",
    "bystro.cas",
    "kreditavan",
    "blago cash",
    "ultracredi",
    "e-cash",
    "creditmax",
    "money_poin",
    "icases.ua",
    "dengi24",
    "hit_cash",
    "credit ok",
    "intercash",
    "monese",
    "elitcredit",
    "ify.credit",
    "dorcas",
    "moneyglad",
    "ifycredit",
    "uakredit",
    "bystro.cash",
    "kreditor",
    "dengidengi",
    "cashbe",
    "micredit",
    "casemaniac",
    "crazy case",
    "silver_cas",
    "sens credit",
    "mlcrocredi",
    "casey",
    "atom case",
    "cascata",
    "cashbox",
    "luckycash",
    "creditavir",
    "glad4money",
    "casekey",
    "kredit vsem",
    "multicast",
    "ultracashua",
    "hypno-casa",
    "avanscredit",
    "money_help",
    "casofficial",
    "incredo",
    "jeanscasual",
    "4-cases",
    "cool credit",
    "simplemone",
    "moneyup",
    "unicredit",
    "secretcase",
    "silvermone",
    "moneyjar",
    "nillkincase",
    "oncredit",
    "pancredit",
    "the-credit",
    "kredit_112",
    "flashcash",
    "creditbot",
    "creditorxx",
    "money_flas",
    "elitcases",
    "money_flash",
    "globalmone",
    "selectmoney",
    "prostomone",
    "kredenscafe",
    "moneyvalue",
    "sens credi",
    "money_star",
    "1case",
    "credit4u",
    "uitracredi",
    "mistermoney",
    "credit bot",
    "micrediton",
    "bit_money",
    "monet home",
    "tuskcasino",
    "casofficia",
    "prostomoney",
    "ultracredit",
    "reficredit",
    "ab.case",
    "money club",
    "moneysend",
    "creditnice1",
    "creditlite",
    "personcase",
    "bananacred",
    "likemoney",
    "creditavira",
    "pocketmone",
    "new_money",
    # -------- TEST Datatset ----------
    "cash.ua",
    "cashalot",
    "money.space",
    "incasso",
    "mistermone",
    "kredit-0",
    "zaxidkredit",
    "skymoney",
    "veo_cash",
    "kredit vse",
    "starcash",
    "simplemoney",
    "casual",
    "happycredit",
    "pocketmoney",
    "easy money",
    "takeurmoney",
    "888casino",
    "shipmoney",
    "mobilcase",
    "ify-credit",
    "foodpicasso",
    "silvermoney",
    "maxicredit",
    "vivid mone",
    "bezcredito",
    "squadcast",
    "mcmoney",
    "silver_cash",
    "case2case",
    "globalmoney",
    "nillkincas",
    "lucyscasino",
    "kredo-shop",
    "whitecredi",
    "micash",
    "cashpro",
    "cash4u",
    "kreditsous",
    "docassist",
    "bezcreditov",
    "dengivdolg",
    "whitecredit",
    "nicecredit",
    "creditnice",
    "microcredit",
    "youcash",
    "money_point",
    "credltmone",
    "pumb online",
]


food_delivery_companies = [
    "dominos",
    "sushi wok",
    "sushimaster",
    "sushi icons",
    "pizza 33",
    "sushimaste",
    "sushimaste",
    "glovo",
    "budusushi",
    "smilefood",
    "sushimaster",
    "sushi-poin",
    "sushi boss",
    "sushimaste",
    "sushimaste",
]


health_companies = [
    "e-health",
    "apteka911",
    "synevo",
    "helsi",
    "medcard24",
    "24/7 лікар",
    "med-servic",
    "med-service",
    "medcity.ua",
    "liki24.com",
    "apteka24.ua",
    "aptekanetu",
    "apteka d.s.",
    "podorozhnyk",
    "med-servlc",
    "apteka nc",
    "podorozhny",
    "med-servlce",
    "likar.info",
    "medcity",
    "likar",
    "apteka d.s",
    "leleka",
]

delivery_companies = [
    "novaposhta",
    "ukrposhta",
    "meestua",
    "global24",
]

competitors_companies = [
    "kyivstar",
    "vodafone ua",
    "lifecell",
    "kyivdigital",
]

competitors_provider_companies = [
    "ukrtelecom",
    "triolan",
    "datagroup",
    "langate",
    "fregat.com",
    "volia",
    "vega",
    "viasat",
]

shops_companies = [
    "rozetka",
    "prom.ua",
    "allo",
    "olx",
    "makeup",
    "citrus.ua",
    "stylus",
    "eldorado",
    "comfy",
    "foxtrot.ua",
    "epicentrk",
    "colins",
    "eva",
    "yakaboo",
    "exist.ua",
    "metro",
    "eva-mozayk",
    "mycredit.ua",
    "epic games",
    "bonjour",
    "citrus",
    "dzvlnok",
    "domino's",
    "fast box",
    "gift-servi",
    "kids-room",
    "kioto",
    "lemon.box",
    "link.dating",
    "mall",
    "medav",
    "ohlala",
    "photo-room",
    "pond",
    "rozetka, магазин",
    "shopster",
    "sofa_dream",
    "stay.cafe",
    "студия",
    "студия.меб",
    "ушастик",
    "холдинг",
    "цифровой",
    "шары",
    "shop_zakaz",
    "top-shop",
    "віці",
    "інтернет",
    "юа_магазин",
]

messengers_companies = [
    "viber",
    "whatsapp",
    "telegram",
    "facebook",
    "google",
    "instagram",
    "tiktok",
    "snapchat",
    "linkedin",
    "zoom",
    "discord",
    "twitter",
]

cyberpolice_companies = [
    "cyberpolice",
    "cyberpolic",
]

vodafone_support = [
    "111",
]

vodafone_survey = [
    "273",
    "275",
    "277",
]

vodafone_new_customer = [
    "222",
]

vodafone_services = [
    "30094",
    "7777",
    "30094",
    "5010",
    "1020",
    "2828",
    "30042",
    "vodafone u",
]

fraud_numbers = [
    "380800305555",
    "380442020202",
    "380444950405",
    "380445276363",
    "380445370222",
    "380442220333",
    "380442009010",
    "380442204404",
    "380442901988",
    "380567878104",
    "380666163133",
    "380677880884",
    "380957321212",
    "380958788181",
    "380443519998",
    "380995055757",
    "380800210800",
]

bank_numbers = [
    "380800500500",
    "380800500850",
    "380800307010",
    "380800502050",
    "380800504450",
    "380800502030",
    "380800507700",
    "380800504400",
    "380800309000",
    "380800307030",
    "380442907290",
    "380443630133",
    "380444908888",
    "380444900500",
    "380962907290",
    "729",
    "3700",
]

competitors_numbers = [
    "380674660466",
    "380997344444",
    "380505022250",
]

rescue_numbers = [
    "112",
    "102",
    "103",
    "dsns ukr",
]

verify_numbers = [
    "verify",
]

phone_companies = [
    "xiaomi",
    "apple",
    "samsung",
    "huawei",
]


bnum_categories_dict = {
    "bank_companies": bank_companies,
    "work_companies": work_companies,
    "grocery_companies": grocery_companies,
    "post_companies": post_companies,
    "taxy_companies": taxy_companies,
    "credits_companies": credits_companies,
    "food_delivery_companies": food_delivery_companies,
    "health_companies": health_companies,
    "delivery_companies": delivery_companies,
    "competitors_companies": competitors_companies,
    "competitors_provider_companies": competitors_provider_companies,
    "shops_companies": shops_companies,
    "messengers_companies": messengers_companies,
    "cyberpolice_companies": cyberpolice_companies,
    "vodafone_support": vodafone_support,
    "vodafone_survey": vodafone_survey,
    "vodafone_services": vodafone_services,
    "vodafone_new_customer": vodafone_new_customer,
    "fraud_numbers": fraud_numbers,
    "bank_numbers": bank_numbers,
    "competitors_numbers": competitors_numbers,
    "rescue_numbers": rescue_numbers,
    "verify_numbers": verify_numbers,
    "phone_companies": phone_companies,
}

finance_topic = [
    "bank_companies",
    "bank_numbers",
    "credits_companies",
]

vodafone_topic = [
    "vodafone_support",
    "vodafone_survey",
    "vodafone_services",
    "vodafone_new_customer",
]

casual_topic = [
    "work_companies",
    "grocery_companies",
    "post_companies",
    "taxy_companies",
    "food_delivery_companies",
    "health_companies",
    "delivery_companies",
    "shops_companies",
    "fraud_numbers",
    "rescue_numbers",
    "verify_numbers",
    "phone_companies",
    "cyberpolice_companies",
    "other",
]

competitors_topic = [
    "competitors_companies",
    "competitors_provider_companies",
    "competitors_numbers",
]

messengers_topic = ["messengers_companies"]


bnum_topics = {
    "finance_topic": finance_topic,
    "vodafone_topic": vodafone_topic,
    "casual_topic": casual_topic,
    "competitors_topic": competitors_topic,
    "messengers_topic": messengers_topic,
}


def assign_bnum_category(dataframe, bnum_categories_dict):
    dataframe["bnum_category"] = "other"
    dataframe["bnum_topic"] = "other"

    for category, numbers in bnum_categories_dict.items():
        dataframe["bnum_category"] = np.where(
            dataframe["bnum"].isin(numbers), category, dataframe["bnum_category"]
        )

    for category, numbers in bnum_topics.items():
        dataframe["bnum_topic"] = np.where(
            dataframe["bnum_category"].isin(numbers), category, dataframe["bnum_topic"]
        )

    vodafone_all = ["vodafone_support", "vodafone_survey", "vodafone_services", "vodafone_new_customer"]

    casual = [
        "work_companies",
        "grocery_companies",
        "post_companies",
        "taxy_companies",
        "food_delivery_companies",
        "health_companies",
        "delivery_companies",
        "shops_companies",
        "other",
    ]

    dataframe["vodafone"] = np.where(dataframe["bnum_category"].isin(vodafone_all), True, False)
    dataframe["casual"] = np.where(dataframe["bnum_category"].isin(casual), True, False)

    return dataframe
