import numpy as np
import jieba
from functools import wraps
import time
import os

# a simple timer to print the time elapsed of function
HOUR = 60*60
MINUTE = 60
SECOND = 1


def time_tick(func):
    @wraps(func)
    def call(*args, **kwargs):
        call_begin = time.time()
        rs = func(*args, **kwargs)
        call_end = time.time()
        t = call_end-call_begin
        h, trace = t//(HOUR), t % (HOUR)
        m, s = trace//(MINUTE), trace % (MINUTE)
        elapsed_info = "call {} elapsed time:{}h {}m {:.2f}s".format(
            func.__name__,
            h,
            m,
            s
        )
        print(elapsed_info)
        if rs is not None:
            return rs
    return call


class SentenceEmbdding:
    def __init__(self, w2v_fp: str, user_dict: str = None, **kwargs):
        self.w2v_fp = w2v_fp
        self.fp_seperator = kwargs.get("seperaotr", " ")
        self.user_dict = user_dict
        if self.user_dict is not None:
            jieba.load_userdict(user_dict)
        self.unknown = "</s>"

    @time_tick
    def _load_vecotor(self):
        self.w2v_dict = {}
        with open(self.w2v_fp, mode="r", encoding="utf-8") as f:
            _, dims = f.readline().strip().split(self.fp_seperator)
            self.vector_dim = int(dims)
            for line in f:
                line_split = line.strip().split(self.fp_seperator)
                word, vector = line_split[0], line_split[1:]
                vector = [float(value) for value in vector]
                self.w2v_dict[word] = vector

    def encode(self, sentence, method="average"):
        """
        Args:
            sentences:str,ori text
            method:str,can be min,max,average

        Returns:
            sentence vector
        """
        # you can register your own method here,but shoud have func signature like func(array,axis=-1,**kwargs)
        method_dict = {
            "average": np.mean,
            "min": np.min,
            "max": np.max
        }
        if method not in method_dict:
            raise ValueError("we only support methods for %s,but got %s" % (
                str(list(method_dict.keys())),
                method
            ))
        words = jieba.lcut(sentence)
        vector_list = []
        unknown = self.w2v_dict[self.unknown]
        for word in words:
            v = self.w2v_dict.get(word) or unknown
            vector_list.append(v)
        array= np.array(vector_list)
        func = method_dict[method]
        array = func(array, axis=0)
        value=array.to_list()
        return value


def save_embeddings(fp:str,dst_prefix:str,encoder:SentenceEmbdding,ignore_head=True):
    suffix="text.vec"
    text_saver=os.path.join(dst_prefix,suffix)
    dst_writer=open(text_saver,mode="w",encoding="utf-8")

    def _worker(text,max_keep=5):
        rs=[]
        uk=encoder.w2v_dict[encoder.unknown]
        text_split=text.split("\001")
        for t in text_split:
            v=encoder.encode(t)
            rs.append(v)
        if len(rs)>=max_keep:
            rs=rs[:max_keep]
        elif len(rs)<max_keep:
            for _ in range(max_keep-len(rs)):
                rs.append(uk)
        rs=np.array(rs).flatten().tolist()
        return rs

    src=open(fp,mode="r",encoding="utf-8")
    if ignore_head:
        src.readline()
    for line in  src:
        line_split=line.strip().split("\002")
        text1,text2,text3=line_split
        v1=_worker(text1)
        v2=_worker(text2)
        v3=_worker(text3)
        v1_content="\001".join([str(v) for v in v1])
        v2_content="\001".join([str(v) for v in v2])
        v3_content="\001".join([str(v)for v in v3])
        write_content="\002".join(
            [
                v1_content,v2_content,v3_content
            ]
        )
        dst_writer.write(write_content)
        dst_writer.write("\n")
    dst_writer.close()

if __name__=="__main__":
    encoder=SentenceEmbdding(
        w2v_fp="assets/w2v_ai_session_5month_output_wiki_20.vec"
    )
    encoder._load_vecotor()
    save_embeddings(
        fp="data/test.txt",
        dst_prefix="vector",
        encoder=encoder
    )