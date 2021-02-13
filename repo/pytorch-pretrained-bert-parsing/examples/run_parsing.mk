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
ifdef WORD_SEGMENTATION
	RUN_PARSING_ARGS += --word_segmentation
endif
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
ifdef USE_GOLD_WORD_SEGMENTATION_IN_TEST
	RUN_PARSING_ARGS += --use_gold_segmentation_in_test
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

## pas analysis options
ifdef COREFERENCE
	RUN_PARSING_ARGS += --coreference
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

# pas analysis
ifdef PAS_ANALYSIS
# git clone git@bitbucket.org:ku_nlp/nn_based_anaphora_resolution.git
NN_BASED_ANAPHORA_RESOLUTION_DIR := /somewhere/nn_based_anaphora_resolution

MAX_SEQ_LENGTH := 128
RUN_PARSING_ARGS += --pas_analysis
SPECIAL_TOKENS := 著者,読者,不特定:人,NULL,NA
CASE_STRING := ガ,ヲ,ニ,ガ２
RUN_PARSING_ARGS += --special_tokens $(SPECIAL_TOKENS) --case_string $(CASE_STRING)
PAS_RESULT_TXT := $(OUTPUT_DIR)/result.txt
PAS_RESULT_HTML := $(OUTPUT_DIR)/result.html
PAS_TARGET := test
PAS_EVAL_CORPUS := kwdlc
PAS_EVAL_CORPUS_BASEDIR := /share/tool/nn_based_anaphora_resolution/corpus/$(PAS_EVAL_CORPUS)
KNP_ADD_FEATURE_DIR := $(PAS_EVAL_CORPUS_BASEDIR)/knp_add_feature
DEV_ID_FILE := $(PAS_EVAL_CORPUS_BASEDIR)/dev.files
TEST_ID_FILE := $(PAS_EVAL_CORPUS_BASEDIR)/test.files
TRAIN_FILE := $(PAS_EVAL_CORPUS_BASEDIR)/conll/latest/train.conll
PREDICT_FILE := $(PAS_EVAL_CORPUS_BASEDIR)/conll/latest/$(PAS_TARGET).conll
PREDICTIONS_KC_BASENAME := $(PAS_TARGET)_out.conll
PAS_OUT_KNP_DIR := $(FINE_TUNING_DIR)/$(PAS_TARGET)_out_knp
CONLL2KNP_DONE := $(PAS_OUT_KNP_DIR).done
endif

CUR_DIR := $(PWD)
RUN_PARSING_SCRIPT := $(CUR_DIR)/run_parsing.py
PYTHON_COMMAND := $(shell which python)
PYTHON2_COMMAND := $(PYTHON_COMMAND)

ifdef PAS_ANALYSIS
all: $(PAS_RESULT_HTML)
else
ifdef BP_BASED_EVALUATION
all: $(RESULT_WORD_KC_TXT) $(RESULT_WORD_KWDLC_TXT) $(RESULT_BP_KC_DONE) $(RESULT_BP_KWDLC_DONE)
else
all: $(RESULT_WORD_TXT)
endif
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

### pas analysis
# output html
$(PAS_RESULT_HTML): $(PAS_RESULT_TXT)
	$(PYTHON2_COMMAND) $(NN_BASED_ANAPHORA_RESOLUTION_DIR)/scripts/bert_result2html.py --result_file $< > $@

# scorer
$(PAS_RESULT_TXT): $(CONLL2KNP_DONE)
	$(PYTHON2_COMMAND) $(NN_BASED_ANAPHORA_RESOLUTION_DIR)/scripts/scorer.py --knp_dir $(KNP_ADD_FEATURE_DIR) --dev_id_file $(DEV_ID_FILE) --test_id_file $(TEST_ID_FILE) --system_dir $(PAS_OUT_KNP_DIR) --target $(PAS_TARGET) --inter_sentential --relax_evaluation --not_fix_case_analysis --relax_evaluation_multiple_argument > $@ 2> $@.log

# conll2knp
$(CONLL2KNP_DONE): $(FINE_TUNING_MODEL)
	PYTHONPATH=$(NN_BASED_ANAPHORA_RESOLUTION_DIR)/scripts python $(NN_BASED_ANAPHORA_RESOLUTION_DIR)/scripts/corpus/conll2knp.py --output_dir $(PAS_OUT_KNP_DIR) < $(FINE_TUNING_DIR)/$(PREDICTIONS_KC_BASENAME) && \
	touch $@

