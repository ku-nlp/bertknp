# BERTKNP

BERTKNP is a Japanese dependency parser based on BERT. BERTKNP achieves higher dependency accuracy than KNP by four points.

## Requirements

Install the following tools beforehand.

- [Juman++](https://github.com/ku-nlp/jumanpp)
- [KNP](https://github.com/ku-nlp/knp)

## Preparation

1. You should install the following libraries in a python virtual environment.
    ```bash
    $ pip install repo/pytorch-pretrained-bert-parsing
    $ pip install pyknp
    ```
1. Download and install BERT and BERTKNP models
    ```bash
    $ ./download_and_install_models.sh
    ```
    Instead of this script, you can do the following commands step by step.
    ```bash
    $ wget 'http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://lotus.kuee.kyoto-u.ac.jp/nl-resource/JapaneseBertPretrainedModel/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers.zip&name=Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers.zip'
    $ unzip Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers.zip
    $ ln -s Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers pretrained_model
    $ ( cd pretrained_model && ln -s config.json bert_config.json )
    $ wget http://lotus.kuee.kyoto-u.ac.jp/nl-resource/bertknp/model/bertknp-model-20190901.tar.gz
    $ tar zxvf bertknp-model-20190901.tar.gz
    ```

## How to use

```bash
$ echo "昨日訪れた下鴨神社の参道はかなり暗かった。" | jumanpp -s 1 | bin/bertknp
```

- By default, a dependency tree is output. If you need detailed information, use ``the `-tab` option in the same way as KNP.
- The python in your PATH is used. If you want to use the python in your virtual environment, specify by `-p [python path]`.
- You can use a CPU or a GPU. If you use a GPU and have a limited GPU memory, specify multiple GPUs as follows:
    ```bash
    $ export CUDA_VISIBLE_DEVICES="0,1"
    ```

### Use from pyknp

You can use BERTKNP from [pyknp](https://github.com/ku-nlp/pyknp) just like using KNP from pyknp.

```python
from pyknp import KNP

knp = KNP('path/to/bin/bertknp', option='-p /path/to/venv/for/bertknp/bin/python -tab -pyknp', jumanoption='-s 1')
knp.parse('昨日訪れた下鴨神社の参道はかなり暗かった。')  # returns pyknp.BList
```

## References

柴田知秀, 河原大輔, 黒橋禎夫: BERTによる日本語構文解析の精度向上, 言語処理学会 第25回年次大会, pp.205-208, 名古屋, (2019.3).
http://www.anlp.jp/proceedings/annual_meeting/2019/pdf_dir/F2-4.pdf
