# -*- coding: utf-8 -*-

import xmltodict


template = """<?xml version="1.0" encoding="UTF-8"?>
<ChineseChessRecord Version="1.0">
 <Head>
  <Name>{{name}}</Name>
  <URL />
  <From>{{from}}</From>
  <ContestType />
  <Contest />
  <Round>{{round}}</Round>
  <Group />
  <Table />
  <Date>{{date}}</Date>
  <Site>{{site}}</Site>
  <TimeRule />
  <Red>{{red}}</Red>
  <RedTeam>{{redteam}}</RedTeam>
  <RedTime />
  <RedRating />
  <Black>{{black}}</Black>
  <BlackTeam>{{blackteam}}</BlackTeam>
  <BlackTime />
  <BlackRating />
  <Referee />
  <Recorder />
  <Commentator />
  <CommentatorURL />
  <Creator />
  <CreatorURL />
  <DateCreated />
  <DateModified>{{datemodified}}</DateModified>
  <ECCO>D21</ECCO>
  <RecordType>1</RecordType>
  <RecordKind />
  <RecordResult>0</RecordResult>
  <ResultType />
  <FEN>rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1</FEN>
 </Head>
 <MoveList>
 <Move value="00-00" />
{{body}}
 </MoveList>
</ChineseChessRecord>"""
move_template = """  <Move value="{{move}}" />
"""
end_template = """  <Move value="{{move}}" end="1" />
"""

x_dic = dict(zip('abcdefghi', '012345678'))
y_dic = dict(zip('9876543210', '0123456789'))
x_dic_rev = dict(zip('012345678', 'abcdefghi'))
y_dic_rev = dict(zip('9876543210', '0123456789'))


def cbf2move(one_file):
    """
    cbf移动
    """
    out_move = []
    doc = xmltodict.parse(open(one_file, encoding='utf-8').read())
    moves = [i["@value"] for i in doc['ChineseChessRecord']['MoveList']["Move"] if i["@value"] != '00-00']
    for i in moves:
        x1, y1, x2, y2 = str(i[0]), str(i[1]), str(i[3]), str(i[4])
        out_move.append("{}{}{}{}".format(x_dic_rev[x1], y_dic_rev[y1], x_dic_rev[x2], y_dic_rev[y2]))
    return out_move


class CBF:
    """
    CBF类 -> 交叉双边滤波
    """
    def __init__(self, **meta):
        """
        初始化参数
        """
        self.text = template
        self.body = ""
        for key in meta:
            val = meta[key]
            self.text = self.text.replace("{{" + key + "}}", val)

    def receive_moves(self, moves):
        """
        接收移动动作
        """
        self.body = ""
        for move in moves[:-1]:
            a, b, c, d = move
            move = "{}{}-{}{}".format(x_dic[a], y_dic[b], x_dic[c], y_dic[d])
            self.body += move_template.replace("{{move}}", move)
        a, b, c, d = moves[-1]
        move = "{}{}-{}{}".format(x_dic[a], y_dic[b], x_dic[c], y_dic[d])
        self.body += end_template.replace("{{move}}", move)
        self.text = self.text.replace("{{body}}", self.body)

    def dump(self, filename):
        """
        写入数据
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.text)