import jax
import tensorflow as tf
import numpy as np
from transformers import GPT2TokenizerFast
import itertools


class TFRecordLoader:
    def __init__(self, index_fname, batch_size, parse_fn, map_fn=None, restore_state=None):
        if restore_state is not None:
            self.file_idx = restore_state["file_idx"]
            self.file_idx_init = False
            self.used = restore_state["used"]
        else:
            self.file_idx = 0
            self.file_idx_init = True
            self.used = []

        self.index = open(index_fname).read().splitlines()
        self.clean_index = list(filter(lambda x: x not in self.used, self.index))
        self.bs = batch_size
        # self.seq = sample_size
        self.parse_fn = parse_fn

        if map_fn:
            self.map_fn = map_fn
        else:
            self.map_fn = lambda x: x

        self.sample_fn = self.sample_once()

    def reset(self):
        self.file_idx = 0
        self.file_idx_init = True
        self.used = []

        self.clean_index = list(filter(lambda x: x not in self.used, self.index))
        self.sample_fn = self.sample_once()

    def sample_once(self):
        for i in self.clean_index:
            compression = "ZLIB" if "zstd" in i else ""

            file = tf.data.TFRecordDataset(i, compression_type=compression).map(self.parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
            file = file.apply(tf.data.experimental.dense_to_ragged_batch(np.prod(self.bs), drop_remainder=True))
            file = file.prefetch(10)

            for file_idx, data in enumerate(file):
                data = jax.tree_map(lambda x: x.numpy(), data)
                data = self.map_fn(data)

                if not self.file_idx_init and file_idx <= self.file_idx:
                    if file_idx % 1000 == 0:
                        print(f"skipping to batch {self.file_idx}, currently at {file_idx}")
                    continue
                self.file_idx_init = True
                self.file_idx = file_idx
                yield jax.tree_map(lambda x: x.reshape(self.bs + x.shape[1:]), data)
            self.used.append(i)
            self.file_idx = 0

    # this loops infinitely, use .sample_once to get an iterator for validation
    def get_samples(self):
        try:
            return next(self.sample_fn)
        except StopIteration:
            self.reset()
            return self.get_samples()

    def get_state(self):
        return {
            "used": self.used,
            "file_idx": self.file_idx
        }


class TFRecordNewInputs(TFRecordLoader):
    def __init__(self, index_fname, batch_size, sample_size, restore_state=None):
        def tf_parse(example_proto):
            features = {
                "text": tf.io.VarLenFeature(tf.int64)
            }
            parsed_features = tf.io.parse_single_example(example_proto, features)

            return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features["text"])), tf.uint32)

        super().__init__(index_fname, batch_size, tf_parse, restore_state=restore_state)


def _parse_function(example_proto): # https://zhuanlan.zhihu.com/p/552951305  # XD
    feature_desc = {"input_ids": tf.io.VarLenFeature(tf.int64)}
    example = tf.io.parse_single_example(example_proto, feature_desc)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64: t = tf.cast(t, dtype=tf.int32)
        example[name] = tf.sparse.to_dense(t, default_value=0)
        # example[name] = tf.sparse.to_dense(tf.sparse.reorder(t)) # mesh-transformer-jax
    return example

def shard(data, batch_size=None):  # XD
    return jax.tree_map(lambda x: x.numpy().reshape(batch_size + x.shape[1:]), data)  # mtj

def load_tfrecord_dataset(index_fname, batch_size, seq_len, restore_state=None):  # XD
    # adapted from gpt-neo
    fnames = [index_fname] if index_fname.endswith('.tfrecords') else open(index_fname).read().splitlines()
    ds = tf.data.Dataset.from_tensor_slices(fnames)#.repeat()
    ds = ds.apply(tf.data.TFRecordDataset)
    # fp = index_fname; ds = tf.data.TFRecordDataset(fp)
    # # ds = ds.shuffle(buffer_size=min(1000, len(sequences))) # flaxmodels, https://zhuanlan.zhihu.com/p/552951305
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    # gradient_accumulation_steps = 8
    # mp_size, dp_size = 8, 1
    # train_mbs_per_replica = 2 # train_micro_batch_size_per_gpu in deepspeed
    # train_batch_size = (gradient_accumulation_steps, train_mbs_per_replica * dp_size)
    # seq_len = 80  # max(len(s) for s in sequences) == 78
    # ds = ds.apply(tf.data.experimental.dense_to_ragged_batch(np.prod(self.bs), drop_remainder=True)) # mtj
    ds = ds.padded_batch(batch_size=np.prod(batch_size), padded_shapes={'input_ids': [seq_len]},
                        padding_values={'input_ids': 0}, drop_remainder=True)
    ds = ds.prefetch(10)  # mesh-transformer-jax
    # ds = ds.repeat()  # gpt-neo/inputs.py
    # map shard directly over ds won't work, getting AttributeError: 'Tensor' object has no attribute 'numpy'
    # because inside tf.function?, see e.g.:
    # 1) https://stackoverflow.com/questions/34097281/convert-a-tensor-to-numpy-array-in-tensorflow
    # 2) https://github.com/tensorflow/tensorflow/issues/27519
    # ds = ds.map(partial(shard, batch_size=batch_size), num_parallel_calls=tf.data.AUTOTUNE)
    # matthias-wright/flaxmodels/training/stylegan2/data_pipeline.py
    return map(lambda x: shard(x, batch_size=batch_size), iter(ds))

class TFRecordWIT(TFRecordLoader):
    def __init__(self, index_fname, batch_size, restore_state=None, text_tokens=256):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = "<|endoftext|>"
        self.tokenizer.add_special_tokens({'sep_token': '<|sep|>', 'pad_token': '<|pad|>'})

        def map_fn(example):
            tokenizer = self.tokenizer

            def decode(x):
                return tokenizer(["<|endoftext|>" + i.decode() for i in x])["input_ids"]

            texts = [
                decode(example["context_page_description"]),
                decode(example["context_section_description"]),
                decode(example["caption_reference_description"]),
                decode(example["caption_alt_text_description"]),
                decode(example["caption_attribution_description"]),
            ]

            output = []

            for text, dalle in zip(zip(*texts), example["dalle"]):
                all_text = list(itertools.chain(*text))[-text_tokens+1:]

                all_text += [tokenizer.pad_token_id] * ((text_tokens - 1) - len(all_text))

                assert len(all_text) == text_tokens - 1

                all_tokens = all_text + [tokenizer.sep_token_id] + list(dalle + tokenizer.vocab_size + 1)
                output.append(all_tokens)

            return np.array(output)

        def tf_parse(example_proto):
            features = {
                "page_title": tf.io.FixedLenFeature([], tf.string),
                "section_title": tf.io.FixedLenFeature([], tf.string),
                "hierarchical_section_title": tf.io.FixedLenFeature([], tf.string),
                "caption_reference_description": tf.io.FixedLenFeature([], tf.string),
                "caption_attribution_description": tf.io.FixedLenFeature([], tf.string),
                "caption_alt_text_description": tf.io.FixedLenFeature([], tf.string),
                "mime_type": tf.io.FixedLenFeature([], tf.string),
                "context_page_description": tf.io.FixedLenFeature([], tf.string),
                "context_section_description": tf.io.FixedLenFeature([], tf.string),

                "dalle": tf.io.FixedLenFeature([1024], tf.int64),
            }

            parsed_features = tf.io.parse_single_example(example_proto, features)

            return parsed_features

        super().__init__(index_fname, batch_size, tf_parse, map_fn, restore_state=restore_state)


if __name__ == "__main__":
    # d = TFRecordNewInputs("data/pile.val.index", (8, 32), 2048)
    # for idx, i in enumerate(d.sample_once()):
    #     print(i)
    #     break

    d = TFRecordWIT("data/wit_dalle.train.index", (8, 32))
    for idx, i in enumerate(d.sample_once()):
        print(i)
        break

    print()
