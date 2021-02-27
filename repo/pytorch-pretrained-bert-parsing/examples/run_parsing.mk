OUTPUT_DIR_BASENAME := /larch/shibata/bert/parsing/result
TASK_NAME := 181123_subword_wikipedia
OUTPUT_DIR := $(OUTPUT_DIR_BASENAME)/$(TASK_NAME)
FINE_TUNING_DIR := $(OUTPUT_DIR)/finetuning
FINE_TUNING_MODEL := $(FINE_TUNING_DIR)/pytorch_model.bin
PREDICTIONS_KC_BASENAME := predictions_kc.txt
PREDICTIONS_KC := $(FINE_TUNING_DIR)/$(PREDICTIONS_KC_BASENAME)
PREDICTIONS_KWDLC := $(FINE_TUNING_DIR)/predictions_kwdlc.txt
RESULT_WORD_KC_TXT :=  $(FINE_TUNING_DIR)/result_word_kc.txt
RESULT_WORD_KWDLC_TXT :=  $(FINE_TUNING_DIR)/result_word_kwdlc.txt

RESULT_BP_KC_DONE := $(FINE_TUNING_DIR)/result_bp_kc.done
RESULT_BP_KWDLC_DONE := $(FINE_TUNING_DIR)/result_bp_kwdlc.done

PRETRAINED_MODEL_DIR := /larch/shibata/bert/preprocess/181123_subword_wikipedia/pretraining_model_10e

TRAIN_FILE := /mnt/hinoki/kawahara/work/bert-parsing/train/merged-train.conllu
PREDICT_FILE := /mnt/hinoki/kawahara/work/bert-parsing/test/kc-test-jppv2-const_m1-filter-w_head-v3.conllu
KWDLC_PREDICT_FILE := /mnt/hinoki/kawahara/work/bert-parsing/test/kwdlc-test-jppv2-const_m1-filter-w_head-v3.conllu

NN_TEST_BED_DIR := /home/shibata/work/nn_parser_testbed

RUN_JUMAN_BERTKNP_EVALUATION_SH := /orange/kawahara/work/parse-eval/run-juman-bertknp-evaluation-gpu.sh
BERTKNP := /share/tool/bertknp/bin/bertknp
JUMANPP := /orange/kawahara/share/tool/jumanpp/cmake-build-dir/src/jumandic/jumanpp

RUN_PARSING_ARGS=
ifdef POS_TAGGING
	RUN_PARSING_ARGS += --pos_tagging
endif
ifdef SUBPOS_TAGGING
	RUN_PARSING_ARGS += --subpos_tagging
endif
ifdef FEATS_TAGGING
	RUN_PARSING_ARGS += --feats_tagging
endif
ifdef PARSING
	RUN_PARSING_ARGS += --parsing
endif
ifdef ESTIMATE_DEP_LABEL
	RUN_PARSING_ARGS += --estimate_dep_label
endif
ifdef USE_GOLD_POS_IN_TEST
	RUN_PARSING_ARGS += --use_gold_pos_in_test
endif
ifdef USE_TRAINING_DATA_RATIO
	RUN_PARSING_ARGS += --use_training_data_ratio $(USE_TRAINING_DATA_RATIO)
endif
ifdef LEARNING_RATE
	RUN_PARSING_ARGS += --learning_rate $(LEARNING_RATE)
endif
ifdef NUM_TRAIN_EPOCHS
	RUN_PARSING_ARGS += --num_train_epochs $(NUM_TRAIN_EPOCHS)
endif
ifdef PARSING_ALGORITHM
	RUN_PARSING_ARGS += --parsing_algorithm $(PARSING_ALGORITHM)
endif
ifdef H2Z
	RUN_PARSING_ARGS += --h2z
endif


LANG := ja
ifndef BP_BASED_EVALUATION
PREDICTIONS_KC_BASENAME := predictions.txt
PREDICTIONS_KC := $(FINE_TUNING_DIR)/$(PREDICTIONS_KC_BASENAME)
RESULT_WORD_TXT :=  $(FINE_TUNING_DIR)/result_word.txt
endif

ifeq ($(LANG), zh)
TRAIN_FILE := /home/kawahara/share/tool/nn_parser/data/ctb51j-train.conllu
PREDICT_FILE := /home/kawahara/share/tool/nn_parser/data/ctb51j-test.conllu
endif

UD_EVAL_FILE := conll17_ud_eval.py
GOLD_FILE := $(PREDICT_FILE)

EVAL_ARGS := -v
ifdef CTB
	EVAL_ARGS += --exclude_punctuations
endif

MAX_SEQ_LENGTH := 192
TRAIN_BATCH_SIZE := 16
PREDICT_BATCH_SIZE := 8
GRADIENT_ACCUMULATION_STEPS := 1

CUR_DIR := $(PWD)
RUN_PARSING_SCRIPT := $(CUR_DIR)/run_parsing.py
PYTHON_COMMAND := $(shell which python)
PYTHON2_COMMAND := $(PYTHON_COMMAND)

ifdef BP_BASED_EVALUATION
all: $(RESULT_WORD_KC_TXT) $(RESULT_WORD_KWDLC_TXT) $(RESULT_BP_KC_DONE) $(RESULT_BP_KWDLC_DONE)
else
all: $(RESULT_WORD_TXT)
endif

$(FINE_TUNING_MODEL):
	mkdir -p $(FINE_TUNING_DIR) && LANG=ja_JP.UTF-8 python run_parsing.py $(RUN_PARSING_ARGS) --train_file $(TRAIN_FILE) --bert_model $(PRETRAINED_MODEL_DIR) --output_dir $(FINE_TUNING_DIR) --do_train --max_seq_length $(MAX_SEQ_LENGTH) --predict_file $(PREDICT_FILE) --do_predict --train_batch_size $(TRAIN_BATCH_SIZE) --predict_batch_size $(PREDICT_BATCH_SIZE) --prediction_result_filename $(PREDICTIONS_KC_BASENAME) --gradient_accumulation_steps $(GRADIENT_ACCUMULATION_STEPS) 2> $(FINE_TUNING_DIR)/$(PREDICTIONS_KC_BASENAME).log 

$(PREDICTIONS_KC): $(FINE_TUNING_MODEL)

$(PREDICTIONS_KWDLC): $(FINE_TUNING_MODEL)
	python run_parsing.py $(RUN_PARSING_ARGS) --bert_model $(PRETRAINED_MODEL_DIR) --output_dir $(FINE_TUNING_DIR) --max_seq_length $(MAX_SEQ_LENGTH) --predict_file $(KWDLC_PREDICT_FILE) --do_predict --prediction_result_filename predictions_kwdlc.txt --predict_batch_size $(PREDICT_BATCH_SIZE)

# word-based evaluation
$(RESULT_WORD_KC_TXT): $(PREDICTIONS_KC)
	python $(NN_TEST_BED_DIR)/evaluation_script/$(UD_EVAL_FILE) $(EVAL_ARGS) /orange/kawahara/work/parse-eval/data/kyoto/latest/conllu/test/kc-test.conllu $< > $@

$(RESULT_WORD_KWDLC_TXT): $(PREDICTIONS_KWDLC)
	python $(NN_TEST_BED_DIR)/evaluation_script/$(UD_EVAL_FILE) $(EVAL_ARGS) /orange/kawahara/work/parse-eval/data/kwdlc/latest/conllu/test/kwdlc-test.conllu $< > $@

## chinese
$(RESULT_WORD_TXT): $(PREDICTIONS_KC)
	python $(NN_TEST_BED_DIR)/evaluation_script/$(UD_EVAL_FILE) $(EVAL_ARGS) $(GOLD_FILE) $< > $@

# bp-based evaluation
$(RESULT_BP_KC_DONE): $(FINE_TUNING_MODEL)
	echo -n "$(RUN_PARSING_ARGS) --max_seq_length $(MAX_SEQ_LENGTH) --prediction_result_filename - --do_predict" > $(FINE_TUNING_DIR)/kc_args.txt
	HOME=/home/kawahara $(RUN_JUMAN_BERTKNP_EVALUATION_SH) kyoto $(JUMANPP) $(BERTKNP) "-b $(PRETRAINED_MODEL_DIR) -m $(FINE_TUNING_DIR) -p $(PYTHON_COMMAND) -P $(RUN_PARSING_SCRIPT) -O $(FINE_TUNING_DIR)/kc_args.txt"> $@.txt 2> $@.log && mv out/kyoto $(FINE_TUNING_DIR) && touch $@

$(RESULT_BP_KWDLC_DONE): $(FINE_TUNING_MODEL)
	echo -n "$(RUN_PARSING_ARGS) --max_seq_length $(MAX_SEQ_LENGTH) --prediction_result_filename - --do_predict" > $(FINE_TUNING_DIR)/kwdlc_args.txt
	HOME=/home/kawahara $(RUN_JUMAN_BERTKNP_EVALUATION_SH) kwdlc $(JUMANPP) $(BERTKNP) "-b $(PRETRAINED_MODEL_DIR) -m $(FINE_TUNING_DIR) -p $(PYTHON_COMMAND) -P $(RUN_PARSING_SCRIPT) -O $(FINE_TUNING_DIR)/kwdlc_args.txt" > $@.txt 2> $@.log && mv out/kwdlc $(FINE_TUNING_DIR) && touch $@
