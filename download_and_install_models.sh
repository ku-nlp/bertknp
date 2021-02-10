#!/bin/sh

bert_pretrained_model_zip=Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers.zip
bert_pretrained_model_dir=`basename $bert_pretrained_model_zip .zip`
bertknp_model_tar_gz=bertknp-model-20190901.tar.gz


# make "pretrained_model"

if [ ! -d "$bert_pretrained_model_dir" -o ! -s "$bert_pretrained_model_dir/pytorch_model.bin" ]; then
    # download bert_pretrained_model_zip
    if [ ! -s "$bert_pretrained_model_zip" ]; then
	wget 'http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://lotus.kuee.kyoto-u.ac.jp/nl-resource/JapaneseBertPretrainedModel/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers.zip&name=Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers.zip' -O $bert_pretrained_model_zip
	if [ ! -s "$bert_pretrained_model_zip" ]; then
	    echo "Cannot download $bert_pretrained_model_zip"
	    exit 1
	fi
    fi
    unzip $bert_pretrained_model_zip
fi

# need symlink "pretrained_model"
if [ ! -L pretrained_model ]; then
    ln -s $bert_pretrained_model_dir pretrained_model
fi

# need "pretrained_model/bert_config.json"
if [ ! -f pretrained_model/bert_config.json ]; then
    ( cd pretrained_model && ln -s config.json bert_config.json )
fi


# make "model"

if [ ! -d model -o ! -s "model/latest/pytorch_model.bin" ]; then
    # download bertknp_model_tar_gz
    if [ ! -s "$bertknp_model_tar_gz" ]; then
	wget http://lotus.kuee.kyoto-u.ac.jp/nl-resource/bertknp/model/$bertknp_model_tar_gz
	if [ ! -s "$bertknp_model_tar_gz" ]; then
	    echo "Cannot download $bertknp_model_tar_gz"
	    exit 1
	fi
    fi
    tar zxvf $bertknp_model_tar_gz
fi
