"""
generator contrast smaple pair
"""
import os
import re
import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
import numpy as np
import pandas as pd
import configparser

from tensorflow.python.framework.c_api_util import ScopedTFFunction


def load_rep_cnf(cnf: str) -> dict:
    config_parser = configparser.ConfigParser()
    config_parser.read(cnf)
    replace_dict = {}
    replace_words = config_parser["word replace"]
    for item in replace_words:
        k, v = item.split("=")
        replace_dict[k] = v
    return replace_dict


class PairGenerator(AutoRegressiveDecoder):
    def __init__(self, checkpoint_path, config_path, vocab_path, max_len=128):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.vocab_path = vocab_path
        self.max_len = max_len

        self.tokenizer = Tokenizer(self.vocab_path, do_lower_case=True)
        self.bert = build_transformer_model(
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            model="roformer",
            with_pool="linear",
            application="unilm",
            return_keras_model=False
        )

        self.encoder = keras.models.Model(
            self.bert.model.inputs, self.bert.model.outputs[0])

        self.seq2seq = keras.models.Model(
            self.bert.model.inputs, self.bert.model.outputs[1])

        super().__init__(start_id=None, end_id=self.tokenizer._token_end_id, maxlen=self.max_len)

    @AutoRegressiveDecoder.wraps('probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate(
            [segment_ids, np.ones_like(output_ids)], 1)
        return self.seq2seq.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, n=1, topk=5):
        token_ids, segment_ids = self.tokenizer.encode(
            text, maxlen=self.max_len)
        output_ids = self.random_sample([token_ids, segment_ids], n, topk)
        return [self.tokenizer.decode(ids) for ids in output_ids]

    def gen_synonyms(self, text, n=100, k=20):
        sims = self.generate(text=text, n=n)
        rs=set()
        # rs = [r for r in set(rs) if r != text]
        clean_text=re.subn(re.compile(r"[^a-zA-Z0-9\u4E00-\u9FA5]"),"",text)[0]
        for r in set(sims):
            r=re.subn(re.compile(r"[^a-zA-Z0-9\u4E00-\u9FA5]"),"",r)[0]
            if r!=clean_text:
                # if len(r)-len(clean_text)< 5:
                rs.add(r)
        rs=list(rs)
        token_ids, segment_ids = [], []
        for r in rs:
            t_id, s_id = self.tokenizer.encode(r)
            token_ids.append(t_id)
            segment_ids.append(s_id)
        token_ids = sequence_padding(token_ids)
        segment_ids = sequence_padding(segment_ids)
        Z = self.encoder.predict([token_ids, segment_ids])
        Z /= (Z**2).sum(axis=1, keepdims=True)**0.05
        scores = np.dot(Z[1:], Z[0])
        args_sort = scores.argsort()
        sorted_score=[(rs[i], scores[i]) for i in args_sort[::-1][:k]]
        return sorted_score
    
    def test(self):
        text="你自己打车过来，到时候给你车费"
        sims=self.gen_synonyms(text,n=20)


class contrastPairSample:
    def __init__(self, source_file: str, dst_file: str, data_generator: PairGenerator, run_mode="single",**kwargs):
        self.source_file = source_file
        dst_dirname = os.path.dirname(dst_file)
        if not os.path.exists(dst_dirname):
            os.makedirs(dst_dirname)
        self.dst_file = dst_file

        # 举报文本对应
        self.REPORT_CONTENT_KEY = kwargs.get(
            "report_content_key", "strImpeachSrvParam")
        self.SEPERATOR = kwargs.get("seprator", "\001")
        self.SAVE_SEPERATOR = '\002'
        # match pattern
        self.RULE = re.compile(r"[^a-zA-Z0-9\u4E00-\u9FA5，]")

        # text sim gen
        self.data_generator = data_generator

        # replace dict
        if "replace_word" in kwargs:
            self.replace_dict = load_rep_cnf(cnf=kwargs["replace_word"])
        else:
            self.replace_dict = {}
        self.run_mode=run_mode

    def _load_src_data(self) -> list:
        src_reader = open(self.source_file, mode="r", encoding="utf-8")
        columns = src_reader.readline().strip().split(self.SEPERATOR)
        report_content_list = []
        report_content_idx=-1
        for idx, column in enumerate(columns):
            if column == self.REPORT_CONTENT_KEY:
                report_content_idx = idx
                break
    
        if report_content_idx==-1: #invalie value
            raise ValueError("column %s should be in file,but we got %s" % (
                self.REPORT_CONTENT_KEY,
                str(columns)
        ))

        if self.run_mode=="all":
            for line in src_reader:
                line_split = line.strip().split(self.SEPERATOR)
                if len(line_split)<=report_content_idx:
                    print("invalid data %s"%(line))
                    continue
                report_content = line_split[report_content_idx]

                report_content_list.append(
                    self._load_be_reported(report_content)
                )
            return report_content_list
                
        elif self.run_mode=="single":
            writer=open(self.dst_file,mode="w",encoding="utf-8")
            for line in src_reader:
                line_split = line.strip().split(self.SEPERATOR)
                if len(line_split)<=report_content_idx:
                    print("invalid data %s"%(line))
                    continue
                report_content = line_split[report_content_idx]
                contents=self._load_be_reported(report_content)
                sim1_rs,sim2_rs=[],[]
                for idex,c in enumerate(contents):
                    s1,s2=self._single_sim_contrast_pair(c)
                    sim1_rs.append(s1),sim2_rs.append(s2)
                text1=self.SEPERATOR.join(contents)
                text2=self.SEPERATOR.join(sim1_rs)
                text3=self.SEPERATOR.join(sim2_rs)

                write_content=self.SAVE_SEPERATOR.join(
                    [text1,text2,text3,"\n"],
                )
                writer.write(write_content)
                writer.flush()
            writer.close()
        src_reader.close()

    # read all the report content
    def _load_all(self, report_content: str):
        content_list = report_content.split("|")[1:]
        rs_list = []
        for content_item in content_list:
            content_item: str = re.subn(self.RULE, "", content_item)[0]
            for k, v in self.replace_dict.items():
                content_item.replace(k, v)
            rs_list.append(content_item)
        return rs_list

    # only read reporter
    def _load_reporter(self, report_content: str):
        content_list = report_content.split("|")[1:]
        rs_list = []
        for content_item in content_list:
            if content_item.startswith("举报人:"):
                content_item: str = re.subn(self.RULE, "", content_item)[0]
                for k, v in self.replace_dict.items():
                    content_item.replace(k, v)
                rs_list.append(content_item.replace("举报人:",""))
        return rs_list

    # only read be_reported
    def _load_be_reported(self, report_content: str):
        content_list = report_content.split("|")[1:]
        rs_list = []
        for content_item in content_list:
            if content_item.startswith("举报人:"):
                content_item: str = re.subn(self.RULE, "", content_item)[0]
                for k, v in self.replace_dict.items():
                    content_item.replace(k, v)
                rs_list.append(content_item.replace("被举报人:",""))
        return rs_list

    def _contrast_pair(self, ori_content: str) -> tuple:
        sim_contents = self.data_generator.gen_synonyms(
            text=ori_content,
            n=100,
            k=10
        )
        rs = (sim_contents[0][0], sim_contents[1][0])
        return rs

    def _single_sim_contrast_pair(self,text):
        return self._contrast_pair(text)

    def save_sim_contrast_pair(self):
        contents = self._load_src_data()
        sim1_list, sim2_list = [], []
        for items in contents:
            t1, t2 = [], []
            for item in items:
                sim1, sim2 = self._contrast_pair(item)
                t1.append(sim1)
                t2.append(sim2)
            sim1_list.append(self.SEPERATOR.join(t1))
            sim2_list.append(self.SEPERATOR.join(t2))
        assert len(sim2_list) == len(sim2_list) == len(
            contents), "data size not match"
        df = pd.DataFrame(
            {
                "text1": contents,  # original text
                "text2": sim1_list,  # similarity text1
                "text3": sim2_list  # similarity text2
            }
        )
        # save data to dist
        df.to_csv(
            self.dst_file,
            sep=self.SAVE_SEPERATOR,
            index=False
        )
    def run(self):
        if self.run_mode=="single":
            self._load_src_data()
        elif self.run_mode=="all":
            self.save_sim_contrast_pair()

if __name__ == "__main__":
    pair_generator = PairGenerator(
        checkpoint_path="pretrain/chinese_roformer-sim-char_L-12_H-768_A-12/bert_model.ckpt",
        config_path="pretrain/chinese_roformer-sim-char_L-12_H-768_A-12/bert_config.json",
        vocab_path="pretrain/chinese_roformer-sim-char_L-12_H-768_A-12/vocab.txt",
        max_len=200
    )
    try:
        pair_generator.test()
    except:
        raise RuntimeError("code error when running!")
    
    contrast_pair_sample = contrastPairSample(
        source_file="./data/test.txt",
        dst_file="./contrast_data/test.txt",
        data_generator=pair_generator
    )
    # data gen
    contrast_pair_sample.run()
