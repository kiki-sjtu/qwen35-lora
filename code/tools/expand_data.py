"""Expand the course dataset with auditable rule-generated synthetic examples.

Does not call a language model or train one. Writes to --output, preserving inputs.
"""

from pathlib import Path as _RuntimePath
import sys as _runtime_sys
_runtime_sys.path.insert(0, str(_RuntimePath(__file__).resolve().parents[1]))
from runtime import ensure_project_python
ensure_project_python(__file__)

import argparse
import collections
import hashlib
import json
import math
import random
import re
from pathlib import Path

SEED = 20261029
SYSTEM = '你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。'
FIELDS = {'predicate', 'object_type', 'subject_type', 'object', 'subject'}
CATEGORIES = ['水果', '洗浴', '平板', '酒店', '衣服', '电脑', '书籍', '蒙牛']
REL_PROMPTS = {
 'train': ['抽取以下文本中的SPO关系，返回JSON数组。', '请找出文中的全部明确关系，用JSON列表回答。', '从这段资料抽取关系三元组，并标注实体类型，只输出JSON。', '请将下文包含的关系整理为JSON数组。', '阅读材料，提取主体、关系和客体，按JSON格式返回。', '下面描述包含哪些明确的SPO？输出JSON数组即可。'],
 'dev': ['把材料中有依据的实体关系转成JSON列表，不添加说明。', '识别下列文字所表达的关系，结果必须是JSON数组。', '请以JSON数组列出段落里的SPO以及主客体类型。', '依据这段文字进行关系抽取，仅返回JSON结果。', '整理以下内容中的实体关系，保留五个标准字段并用JSON数组输出。'],
}
CLASS_PROMPTS = {
 'train': ['下面是一条关于什么的评论？用JSON列表回答。', '判断这段评论的主题类别，只输出JSON数组。', '这段评价描述的是什么类别？返回JSON列表。', '请识别下列评论的类别，以JSON数组作答。', '给下面的消费评价分类，用列表形式返回类别名称。'],
 'dev': ['根据评价内容确定评论对象，输出类别的JSON列表。', '这位用户评价的是哪一类商品或服务？只返回JSON数组。', '阅读这段体验描述，以JSON列表标注它的主题。', '下列消费者反馈应归入什么类别？答案采用JSON数组。'],
}

# Split-specific phrase banks. All examples are fictional, mechanically labeled.
PRODUCTS = {
 '水果': ('苹果 梨 葡萄 橙子 桃子 芒果 猕猴桃 草莓 蓝莓 柚子'.split(),
          '刚收到一箱苹果 新买的葡萄 桌上这盒草莓 家里买的橙子 这批猕猴桃 一袋新鲜的梨 这次买的芒果 昨天到的柚子'.split()),
 '洗浴': ('这家洗浴中心 这间澡堂 附近的浴场 这家洗浴会馆 这处温泉洗浴店 新开的洗浴馆 常去的浴池 朋友推荐的澡堂'.split(),
          '周末去的洗浴馆 出差时去的澡堂 住处旁的洗浴会馆 那间温泉浴场 昨晚去的浴池 路边这家洗浴中心 新体验的洗浴店 这次选的洗浴场所'.split()),
 '平板': ('这台平板电脑 新买的平板 这款平板电脑 手里的平板 这部平板电脑 这台安卓平板 家里的平板电脑 用来记笔记的平板'.split(),
          '最近入手的平板电脑 这块学习用的平板 出差带的平板电脑 换新的这台平板 刚拆封的平板电脑 通勤用的平板 这款触屏平板电脑 借来体验的平板'.split()),
 '酒店': ('这家酒店 这间宾馆 预订的酒店 这家商务酒店 入住的旅馆 这家度假酒店 车站附近的酒店 这间连锁酒店'.split(),
          '旅途中住的宾馆 本次入住的酒店 会议安排的酒店 昨晚住的旅馆 这次预订的宾馆 行程里的度假酒店 景区旁边的酒店 朋友订的酒店'.split()),
 '衣服': ('这件衬衫 新买的外套 这条长裤 这件羽绒服 这件卫衣 这条半身裙 这件针织衫 这套运动服 这件短袖 这件风衣'.split(),
          '刚到货的衬衣 这回买的牛仔裤 试穿的连衣裙 送来的羊毛衫 这款防晒外套 这条休闲裤 新入手的夹克 这件春季上衣'.split()),
 '电脑': ('这台笔记本电脑 新买的台式电脑 这款笔记本 这台办公电脑 这台游戏笔记本 这套台式主机 这台轻薄本 家里的台式电脑'.split(),
          '刚换的笔记本电脑 组装好的台式机 出差携带的笔记本 这部办公笔记本 新装的电脑主机 开箱后的笔记本电脑 本次购买的台式电脑 用于编程的电脑'.split()),
 '书籍': ('这本散文集 这本小说 这部纸质词典 新买的历史书 这本摄影教材 这套故事书 这本编程教材 这部人物传记'.split(),
          '刚读完的这本文集 到手的这部小说 新版的这本工具书 借来的历史读物 这套科普图书 收到的纸质画册 这本文学选集 最近看的这本传记'.split()),
 '蒙牛': ('这箱蒙牛纯牛奶 蒙牛的这款酸奶 这盒蒙牛低脂牛奶 这袋蒙牛鲜牛奶 这款蒙牛奶饮品 蒙牛的这款原味酸奶 这提蒙牛牛奶 新买的蒙牛纯牛奶'.split(),
          '最近订的蒙牛鲜奶 冰箱里的蒙牛酸奶 这回收到的蒙牛纯奶 早餐喝的蒙牛牛奶 家里这款蒙牛低脂奶 刚拆开的蒙牛原味酸奶 这批蒙牛乳饮品 新尝试的蒙牛酸奶'.split()),
}

FEATURES = {
 '水果': (
  ['果肉清脆，甜味很自然', '拆箱时每一颗都很完整', '成熟度刚好，洗净后就能吃', '香味很足，汁水也多', '个头均匀，放在果盘里很好看', '外皮有点皱，但里面没有坏', '有几颗被压软了，口感一般', '酸味偏重，还需要再放两天', '新鲜程度不错，孩子吃得很开心', '分量与页面描述一致'],
  ['切开后水分充足，闻起来有清香', '到货时带着果梗，表皮没有明显伤痕', '果核不大，吃起来偏甜', '其中两颗有磕碰，其余都能正常食用', '还没完全熟透，入口略有涩感', '颗粒比较饱满，冷藏后味道更好', '有些大小不齐，好在没有腐烂', '剥皮以后香气明显，家人都说不错']),
 '洗浴': (
  ['淋浴的热水很稳定', '浴池的水温合适', '搓澡师傅动作细致', '更衣柜能正常上锁', '提供的浴巾干净柔软', '休息大厅座位不少', '人多时需要排队等搓澡', '地面有水，走路得小心', '洗完之后身体很放松', '蒸汽房温度偏高'],
  ['冲洗区的喷头出水均匀', '泡池边能找到干净的拖鞋', '洗完澡后有独立的吹风区', '搓背服务比预期认真', '更衣室空气有些闷', '桑拿房里的人比较多', '淋浴隔间的排水不太顺畅', '泡澡区域有人及时清理']),
 '平板': (
  ['用触控笔记笔记很顺手', '屏幕触控反应灵敏', '横屏看课程视频很舒服', '分屏阅读和做笔记都很方便', '电池够我上完一天的课', '拿在手里看漫画不算沉', '安装应用后运行基本流畅', '手写时偶尔会出现笔迹延迟', '配套的保护套支撑角度合适', '长时间看视频后背面有点热'],
  ['电子笔在屏幕上画线没有断触', '触屏翻页时跟手程度不错', '上网课时能同时打开电子讲义', '放进背包后携带起来很轻便', '屏幕边缘偶尔识别不到手指', '平板的阅读模式看文档很清楚', '用它手写批注比手机舒服', '平板外接磁吸键盘后打字方便']),
 '酒店': (
  ['前台办理入住很快', '房间隔音效果不错', '床单和枕套都很干净', '早餐种类比较丰富', '电梯距离客房有点远', '房间空调声音略大', '退房时没有额外收费', '窗帘遮光效果很好', '夜里走廊的脚步声有点明显', '客房热水供应稳定'],
  ['到店后前台很快给了房卡', '住的那间客房采光很好', '夜间隔壁说话声听得比较清楚', '行李可以在退房后寄存', '床垫软硬程度适合我', '保洁每天都会补充房间用品', '预订的房型与实际入住一致', '宾馆大堂能借到雨伞']),
 '衣服': (
  ['上身后肩宽和袖长都合适', '面料摸起来柔软', '洗过一次暂时没有缩水', '缝线整齐，几乎看不到线头', '颜色和图片基本一致', '尺码偏小，建议先看尺寸表', '穿起来透气，不容易闷', '领口的走线有一点歪', '口袋深度够用', '衣料比想象中薄一些'],
  ['试穿时腰围正好，不勒肚子', '布料接触皮肤没有刺痒感', '衣摆长度比展示图稍短', '水洗后颜色没有明显变化', '袖口的针脚很细密', '肩部剪裁略紧，抬手不够自在', '衣服挂起来垂坠感不错', '裤脚收边平整，没有多余线头']),
 '电脑': (
  ['开机进入系统的速度很快', '处理文档和表格很流畅', '运行编程软件时没有明显卡顿', '键盘手感适合长时间打字', '风扇在高负载时声音比较大', '硬盘容量足够日常办公', '内存升级后多开程序更稳定', '接上显示器后画面清晰', '机身接口基本满足需求', '游戏时显卡温度略高'],
  ['重启以后系统很快就能使用', '同时打开几个开发工具也没有卡住', '主机里的散热风道比较合理', '长时间编译代码时能听到风扇声', '笔记本触控板定位比较准确', '电脑连接投影仪很顺利', '运行桌面软件时占用的内存较多', '台式机的后置接口数量够用']),
 '书籍': (
  ['纸张厚实，翻页手感不错', '文字排版疏密合适', '印刷清晰，没有明显错页', '内容讲解循序渐进', '装订牢固，翻了很多次也没散', '书角在运输中有一点折痕', '目录安排方便查阅', '读到后半部分仍然很有兴趣', '插图和正文配合得很好', '个别段落需要反复阅读'],
  ['章节划分清楚，找内容很方便', '纸页没有透墨，字迹边缘清楚', '书脊胶装比较结实', '故事的人物刻画细腻', '封面的覆膜有一处小划痕', '读完之后还想再看一遍', '附录索引对查资料很有帮助', '这本纸质书的字号比较适中']),
 '蒙牛': (
  ['奶香味比较自然', '包装上的生产日期很近', '早餐配面包喝很方便', '冷藏之后口感更顺滑', '甜度适中，不会觉得腻', '盒口容易打开', '这一箱每盒都没有漏液', '配料表标注得很清楚', '家里人喝完没有不适', '这款酸奶质地比较浓稠'],
  ['入口的乳香清淡，没有明显异味', '独立奶盒放进包里比较方便', '这批的保质日期标识清晰', '杯装酸奶搅拌后细腻均匀', '喝起来没有想象中那么甜', '奶盒边角完整，没有渗漏', '倒进杯里颜色均匀', '早餐加热后奶味仍然不错']),
}

NAMES = {
 'train': ('赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张', '清远安宁若言景川知夏明月星河云舟书禾松竹海山青南雨林'),
 'dev': ('邓谢邹喻柏窦章苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任', '佑棠锦昭予澄栖晚凌霁怀墨榆珂晏初陌洛灿瑾桐谦枫铃棣'),
}
TITLE_PARTS = {
 'train': ('晨暮晴碧青白远近春夏秋冬松竹梅兰山海江河星月风云晴雨朝夕古新浮流', '光影城桥岛林原谷港岸川田岭路亭苑渡湖泉浦雪沙舟帆笺歌梦语行书', '记集录篇卷传行声调景图寻约书梦曲影画信灯路人时季章辞集说间闻'),
 'dev': ('栖晚霁雾岚澜溯汐潮霜荧灿锦绣绯绀苍翠玄素鹭雁鹤鸥隼鹞鸿鸢鸾鹫鹰雀', '巷坊陌径埠堤陂汀泽渚洲屿崖峡岬峰岫洞壑潭溪涧湫湾礁砾矶岑坞岗庐庵', '札谣赋谱迹章序引简册志问答忆续话章题铭咏赞叙论述吟抄编选读览纪'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def output_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def sample(prompt, body, answer):
    return {'messages': [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': prompt + '\n\n' + body},
        {'role': 'assistant', 'content': json.dumps(answer, ensure_ascii=False)},
    ]}


def normalized(value):
    return re.sub(r'[\W_]+', '', value).casefold()


def body(row):
    return row['messages'][1]['content'].split('\n\n', 1)[-1]


def task(row):
    answer = json.loads(row['messages'][-1]['content'])
    return 'relation' if answer and isinstance(answer[0], dict) else 'classification'


class Entities:
    def __init__(self, split):
        self.split, self.counter = split, 0

    def get(self, kind):
        index = self.counter
        self.counter += 1
        if kind == '人物' or kind == '历史人物':
            surnames, chars = NAMES[self.split]
            capacity = len(surnames) * len(chars) ** 2
            assert index < capacity
            return surnames[index % len(surnames)] + chars[(index // len(surnames)) % len(chars)] + chars[(index // (len(surnames) * len(chars))) % len(chars)]
        a, b, c = TITLE_PARTS[self.split]
        assert index < len(a) * len(b) * len(c)
        word = a[index % len(a)] + b[(index // len(a)) % len(b)] + c[(index // (len(a) * len(b))) % len(c)]
        suffix = {'地点': '镇', '国家': '国', '城市': '市', '行政区': '县', '景点': '景区',
                  '学校': '学院', '企业': '科技有限公司', '机构': '研究中心', '出版社': '出版社',
                  '网站': '小说网', '目': '目', '生物': '虫', '气候': '性气候'}.get(kind, '')
        return word + suffix


def relation_spec(predicate, subject_type, object_type, entities, rng):
    s = entities.get(subject_type)
    if object_type == 'Date':
        o = f'{rng.randrange(1970, 2025)}年{rng.randrange(1, 13)}月{rng.randrange(1, 29)}日'
    elif predicate == '身高':
        o = f'{rng.randrange(150, 198)}厘米'
    elif predicate == '海拔':
        o = f'{rng.randrange(150, 4500)}米'
    elif predicate == '注册资本':
        o = f'{rng.randrange(10, 5000)}万元'
    elif predicate == '人口数量':
        o = f'{rng.randrange(10, 250)}万人'
    elif predicate in ('面积', '占地面积'):
        o = f'{rng.randrange(10, 1000)}' + ('平方千米' if predicate == '面积' else '平方米')
    elif predicate == '民族':
        o = rng.choice(['汉族', '回族', '满族', '壮族', '苗族', '土家族'])
    elif predicate == '朝代':
        o = rng.choice(['唐朝', '宋朝', '明朝', '清朝', '元朝'])
    elif predicate == '气候':
        o = rng.choice(['温带海洋性气候', '热带季风气候', '亚热带季风气候', '温带大陆性气候'])
    else:
        o = entities.get(object_type)
    rel = {'predicate': predicate, 'object_type': object_type, 'subject_type': subject_type, 'object': o, 'subject': s}
    return rel


def clauses(rel, split):
    s, o, p = rel['subject'], rel['object'], rel['predicate']
    # The two banks express identical relation schemas with separate wording.
    patterns = {
     '主演': ('《{s}》由{o}主演', '{o}在影视作品《{s}》中担任主演'),
     '导演': ('《{s}》的导演是{o}', '{o}执导了影视作品《{s}》'),
     '编剧': ('《{s}》的编剧为{o}', '影视作品《{s}》由{o}编写剧本'),
     '制片人': ('《{s}》的制片人是{o}', '{o}担任影视作品《{s}》的制片人'),
     '出品公司': ('《{s}》由{o}出品', '影视作品《{s}》的出品方为{o}'),
     '上映时间': ('《{s}》于{o}上映', '{o}是影视作品《{s}》的上映日期'),
     '改编自': ('《{s}》改编自作品《{o}》', '作品《{o}》被改编成影视作品《{s}》'),
     '作者': ('图书作品《{s}》的作者是{o}', '{o}创作了图书作品《{s}》'),
     '出版社': ('《{s}》由{o}出版', '{o}负责书籍《{s}》的出版发行'),
     '连载网站': ('网络小说《{s}》在{o}连载', '{o}是网络小说《{s}》的连载平台'),
     '主角': ('网络小说《{s}》的主角叫{o}', '{o}是网络小说《{s}》中的主角'),
     '歌手': ('歌曲《{s}》由{o}演唱', '{o}是歌曲《{s}》的演唱者'),
     '作词': ('歌曲《{s}》由{o}作词', '{o}为歌曲《{s}》撰写歌词'),
     '作曲': ('歌曲《{s}》的作曲者是{o}', '歌曲《{s}》的曲作者为{o}'),
     '所属专辑': ('歌曲《{s}》收录在专辑《{o}》中', '专辑《{o}》收录了歌曲《{s}》'),
     '出生日期': ('{s}出生于{o}', '{o}是{s}的出生日期'),
     '出生地': ('{s}的出生地是{o}', '{s}在{o}出生'),
     '国籍': ('{s}的国籍为{o}', '{s}拥有{o}国籍'),
     '民族': ('{s}属于{o}', '{s}的民族是{o}'),
     '毕业院校': ('{s}毕业于{o}', '{o}是{s}的毕业院校'),
     '身高': ('{s}身高{o}', '{s}的身高记录为{o}'),
     '祖籍': ('{s}的祖籍在{o}', '{s}祖籍{o}'),
     '妻子': ('{s}的妻子是{o}', '{o}是{s}的妻子'),
     '丈夫': ('{s}的丈夫是{o}', '{o}是{s}的丈夫'),
     '父亲': ('{s}的父亲是{o}', '{o}是{s}的父亲'),
     '母亲': ('{s}的母亲是{o}', '{o}是{s}的母亲'),
     '朝代': ('历史人物{s}生活于{o}', '{s}是{o}时期的历史人物'),
     '字': ('历史人物{s}，字{o}', '历史人物{s}的字为{o}'),
     '号': ('历史人物{s}，号{o}', '历史人物{s}的号为{o}'),
     '成立日期': ('{s}成立于{o}', '{o}是{s}的成立日期'),
     '总部地点': ('{s}的总部位于{o}', '{o}是{s}的总部所在地'),
     '创始人': ('{s}由{o}创办', '{o}是{s}的创始人'),
     '注册资本': ('{s}的注册资本为{o}', '{s}登记的注册资本是{o}'),
     '占地面积': ('{s}占地面积为{o}', '{s}的用地面积记录为{o}'),
     '简称': ('{s}的简称是{o}', '{s}通常简称为{o}'),
     '所在城市': ('{s}位于{o}', '{o}是景点{s}所在的城市'),
     '首都': ('{s}的首都是{o}', '{o}是{s}的首都'),
     '海拔': ('{s}的海拔为{o}', '{s}位于海拔{o}处'),
     '面积': ('{s}的面积为{o}', '{s}行政区域的面积是{o}'),
     '人口数量': ('{s}的人口数量为{o}', '{s}统计的人口总量为{o}'),
     '气候': ('{s}属于{o}', '{s}的气候类型是{o}'),
     '目': ('{s}在分类上属于{o}', '生物{s}所属的目为{o}'),
     '主持人': ('综艺节目《{s}》的主持人是{o}', '{o}主持电视综艺《{s}》'),
     '嘉宾': ('综艺节目《{s}》的嘉宾是{o}', '{o}以嘉宾身份参与电视综艺《{s}》'),
    }
    if p not in patterns:
        raise ValueError(f'Unhandled relation {p}')
    return patterns[p][split == 'dev'].format(s=s, o=o)


def spouse_inverse(rel):
    if rel['predicate'] not in ('妻子', '丈夫'):
        return None
    return {'predicate': '丈夫' if rel['predicate']=='妻子' else '妻子',
            'subject_type': '人物', 'object_type': '人物',
            'subject': rel['object'], 'object': rel['subject']}


def make_relations(count, schemas, split, rng, entities):
    result = []
    # Cover all observed schemas, with varied 1--4 relations per paragraph.
    for i in range(count):
        number = rng.choices([1, 2, 3, 4], [45, 30, 20, 5])[0]
        selected = [schemas[i % len(schemas)]]
        selected += rng.sample([x for x in schemas if x != selected[0]], number - 1)
        relations = [relation_spec(*schema, entities, rng) for schema in selected]
        sentences = [clauses(rel, split) for rel in relations]
        # Swap mention order sometimes, preserving a complete extraction label set.
        order = list(range(number)); rng.shuffle(order)
        text = '。'.join(sentences[j] for j in order) + '。'
        if i % 5 == 0:
            text += ('这段资料已经整理完毕，具体说明以文中记录为准。' if split == 'train' else '上述内容来自同一份整理材料，本段没有补充其他关系。')
        answer = []
        for j in order:
            answer.append(relations[j])
            inverse = spouse_inverse(relations[j])
            if inverse:
                answer.append(inverse)
        row = sample(rng.choice(REL_PROMPTS[split]), text, answer)
        # Subject/object spans must occur literally; each triple comes from its clause.
        assert all(r['subject'] in text and r['object'] in text for r in answer)
        result.append((row, {'source_type': 'synthetic', 'generator': 'explicit_relation_clauses_v1',
                            'task': 'relation', 'schemas': [list(x) for x in selected]}))
    return result


def make_classifications(count_by_label, split, rng, forbidden_bodies):
    result = []
    for label, count in count_by_label.items():
        objects = PRODUCTS[label][split == 'dev']
        features = FEATURES[label][split == 'dev']
        openings = (['这次体验后想说，', '用了之后来反馈一下，', '整体感受是，', '先说说实际感受，', '简单记录一下：', '给大家提供一个参考，', '没有只看图片，实际体验是：', '用过一段时间后觉得，'] if split == 'train' else
                    ['补充一条自己的评价：', '我的真实使用感受如下：', '这次体验让我印象最深的是，', '拿到后认真体验了一下，', '从我这边的情况来看，', '回头说说这次的体验，', '写点实际反馈供参考：', '对这次选择的感受是，'])
        endings = (['总体上符合预期。', '也有需要改进的地方。', '是否合适还是要看个人需求。', '下次会再考虑。', '这一点对我比较重要。', '希望之后能保持这个水平。', '目前先这样评价。', '购买前最好多了解一下。'] if split == 'train' else
                   ['以上仅是我这一次的感受。', '综合体验还算满意。', '后续有新感受再来补充。', '实际情况大致就是这样。', '想选同类的朋友可以做个参考。', '目前看来没有特别大的问题。', '细节还可以继续改进。', '这次就先记录到这里。'])
        made, attempts = 0, 0
        while made < count:
            attempts += 1
            if attempts > count * 50 + 1000:
                raise RuntimeError(f'Not enough unique classification examples: {label}')
            a, b = rng.sample(features, 2)
            text = rng.choice(openings) + rng.choice(objects) + '，' + a + '，' + b + '。' + rng.choice(endings)
            key = normalized(text)
            if key in forbidden_bodies:
                continue
            forbidden_bodies.add(key)
            row = sample(rng.choice(CLASS_PROMPTS[split]), text, [label])
            result.append((row, {'source_type': 'synthetic', 'generator': 'category_review_composition_v1',
                                 'task': 'classification', 'category': label}))
            made += 1
    return result


def validate(project, datasets, provenance, originals, schemas):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    model = project / 'models' / 'Qwen3.5-0.8B'
    tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
    tokenizer.no_truncation(); tokenizer.no_padding()
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def error(message): raise ValueError(message)
    env.globals['raise_exception'] = error
    template = env.from_string((model / 'chat_template.jinja').read_text(encoding='utf-8'))
    report = {}; keys = {}; entity_sets = {}
    schema_set = set(schemas)
    for split, rows in datasets.items():
        assert len(rows) == len(provenance[split])
        seen, inputs, contents, lengths, synthetic_entities = set(), set(), set(), [], set()
        classification = collections.Counter(); relations = collections.Counter(); counts = collections.Counter()
        original_lines = set()
        for row, meta in zip(rows, provenance[split]):
            messages = row['messages']
            assert [m['role'] for m in messages] == ['system', 'user', 'assistant']
            assert all(isinstance(m['content'], str) and m['content'].strip() for m in messages)
            content = body(row); answer = json.loads(messages[-1]['content'])
            assert isinstance(answer, list) and answer
            if isinstance(answer[0], dict):
                for rel in answer:
                    assert set(rel) == FIELDS
                    if meta['source_type'] == 'synthetic':
                        assert (rel['predicate'], rel['subject_type'], rel['object_type']) in schema_set
                        assert rel['subject'] in content and rel['object'] in content
                        inverse = spouse_inverse(rel)
                        assert clauses(rel, split) in content or (inverse and clauses(inverse, split) in content)
                        synthetic_entities.add(rel['subject'])
                        if rel['object_type'] not in ('Number', 'Date', 'Text', '气候'):
                            synthetic_entities.add(rel['object'])
                    relations[rel['predicate']] += 1
                kind = 'relation'
            else:
                assert len(answer) == 1 and answer[0] in CATEGORIES
                classification[answer[0]] += 1
                kind = 'classification'
            assert meta['task'] == kind
            counts[(meta['source_type'], kind)] += 1
            exact = json.dumps(row, ensure_ascii=False, sort_keys=True)
            key = normalized(messages[1]['content'])
            content_key = normalized(content)
            assert exact not in seen and key not in inputs and content_key not in contents, (split, meta)
            seen.add(exact); inputs.add(key); contents.add(content_key)
            if meta['source_type'] == 'original':
                index = meta['source_line'] - 1
                assert row == originals[split][index]
                original_lines.add(index)
            text = template.render(messages=messages, add_generation_prompt=False, enable_thinking=False)
            assert text.endswith(messages[-1]['content'].strip() + '<|im_end|>\n')
            lengths.append(len(tokenizer.encode(text, add_special_tokens=False).ids))
        assert original_lines == set(range(len(originals[split])))
        keys[split] = {'exact': seen, 'prompt': inputs, 'body': contents}
        entity_sets[split] = synthetic_entities
        ordered = sorted(lengths)
        report[split] = {'rows': len(rows), 'counts': {f'{a}_{b}': n for (a,b),n in counts.items()},
                         'classification_categories': dict(classification), 'relation_triples': dict(relations),
                         'min_tokens': min(lengths), 'max_tokens': max(lengths),
                         'p95_tokens': ordered[math.ceil(len(ordered)*.95)-1],
                         'above_1024_tokens': sum(n>1024 for n in lengths),
                         'above_2048_tokens': sum(n>2048 for n in lengths),
                         'all_original_rows_preserved': True}
    overlap = {k: len(keys['train'][k] & keys['dev'][k]) for k in keys['train']}
    assert not any(overlap.values()), overlap
    entities_overlap = entity_sets['train'] & entity_sets['dev']
    assert not entities_overlap, entities_overlap
    report['cross_split_overlap'] = overlap
    report['synthetic_named_entity_overlap'] = len(entities_overlap)
    report['validation_scope'] = 'Exact/normalized prompts, source bodies, synthetic names, JSON labels, template rendering, token lengths. Not a semantic near-duplicate classifier or a factual audit of original labels.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source', type=Path, help='Original 902/122 dataset directory, e.g. the pre-expansion backup')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Keep feature clauses compatible with every product subtype in each bank.
    replacements = {
      '衣服': {
        '上身后肩宽和袖长都合适': '穿上后松紧程度都合适',
        '领口的走线有一点歪': '有一处走线有一点歪',
        '试穿时腰围正好，不勒肚子': '试穿时尺寸正好，不会紧绷',
        '衣摆长度比展示图稍短': '实际长度比展示图稍短',
        '袖口的针脚很细密': '边缘的针脚很细密',
        '肩部剪裁略紧，抬手不够自在': '剪裁略紧，活动不够自在',
        '衣服挂起来垂坠感不错': '挂起来的垂坠感不错',
        '裤脚收边平整，没有多余线头': '收边平整，没有多余线头'},
      '电脑': {'笔记本触控板定位比较准确': '键盘输入的反应比较及时',
             '台式机的后置接口数量够用': '这台电脑的接口数量够用'},
      '蒙牛': {'盒口容易打开': '封口容易打开',
             '这一箱每盒都没有漏液': '收到的每一份都没有漏液',
             '这款酸奶质地比较浓稠': '营养成分信息标注清楚',
             '杯装酸奶搅拌后细腻均匀': '饮用时口感细腻均匀',
             '奶盒边角完整，没有渗漏': '外包装完整，没有渗漏',
             '独立奶盒放进包里比较方便': '独立包装携带比较方便'},
      '书籍': {'故事的人物刻画细腻': '文字表述容易理解'},
    }
    for category, pairs in replacements.items():
        for bank in FEATURES[category]:
            bank[:] = [pairs.get(phrase, phrase) for phrase in bank]
    project, output = args.project.resolve(), args.output.resolve()
    if output == project / 'data':
        raise ValueError('Generate into a staging directory; do not overwrite source data')
    source = args.source.resolve() if args.source else project/'data'
    originals = {split: load(source/f'{split}.jsonl') for split in ('train', 'dev')}
    assert len(originals['train']) == 902 and len(originals['dev']) == 122, 'Expected original 902/122 dataset; do not recursively expand synthetic data'
    # Infer label ontology from training data only; validation labels are not training seeds.
    schemas = sorted({(r['predicate'],r['subject_type'],r['object_type']) for row in originals['train']
                      if task(row)=='relation' for r in json.loads(row['messages'][-1]['content'])})
    forbidden_bodies = {normalized(body(row)) for rows in originals.values() for row in rows}
    datasets, provenance = {}, {}
    for split, total in [('train', 5000), ('dev', 2000)]:
        rng = random.Random(SEED + (0 if split == 'train' else 1))
        relation_total = round(total * .55)
        classification_total = total - relation_total
        orig_rel = sum(task(row)=='relation' for row in originals[split])
        existing_categories = collections.Counter(json.loads(row['messages'][-1]['content'])[0] for row in originals[split] if task(row)=='classification')
        desired = {label: classification_total//len(CATEGORIES) + (i<classification_total%len(CATEGORIES)) for i,label in enumerate(CATEGORIES)}
        needed = {label: desired[label] - existing_categories[label] for label in CATEGORIES}
        assert all(n >= 0 for n in needed.values())
        entries = [(row, {'source_type': 'original', 'task': task(row), 'source_file': f'{split}.jsonl', 'source_line': i}) for i,row in enumerate(originals[split],1)]
        entries += make_relations(relation_total-orig_rel, schemas, split, rng, Entities(split))
        entries += make_classifications(needed, split, rng, forbidden_bodies)
        assert len(entries)==total
        rng.shuffle(entries)
        datasets[split] = [x[0] for x in entries]
        provenance[split] = [dict(x[1], row_number=i, split=split, seed=SEED) for i,x in enumerate(entries,1)]
    validation = validate(project, datasets, provenance, originals, schemas)
    output.mkdir(parents=True, exist_ok=True)
    for split in ('train','dev'):
        output_jsonl(output/f'{split}.jsonl', datasets[split])
        output_jsonl(output/f'{split}_provenance.jsonl', provenance[split])
        assert load(output/f'{split}.jsonl') == datasets[split]
    # Keep the untouched course validation split available for separate evaluation.
    (output/'dev_original.jsonl').write_bytes((source/'dev.jsonl').read_bytes())
    report = {'seed': SEED, 'method': 'Rule-based controlled synthetic text; no LLM used',
              'original_source_sha256': {split:sha(source/f'{split}.jsonl') for split in ('train','dev')},
              'requested_sizes': {'train':5000,'dev':2000}, 'training_started':False,
              'new_rows':{'train':4098,'dev':1878}, 'validation':validation,
              'limitations': ['Synthetic labels describe fictional text, not independently collected real-world examples.',
                             'Synthetic validation shares relation schemas and generator families with training despite disjoint names and sentence banks.',
                             'The expanded dev is mostly synthetic and is not an independent real-world test set; evaluate dev_original separately.',
                             'Original labels have been preserved without a factual correctness audit.'],
              'output_sha256':{f'{split}.jsonl':sha(output/f'{split}.jsonl') for split in ('train','dev')}}
    (output/'expansion_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
