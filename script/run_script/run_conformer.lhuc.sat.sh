#!/usr/bin/env bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
source /opt/share/etc/gcc-5.4.0.sh
cd /project_bdda6/bdda/jjdeng/espnet/egs/swbd/asr2

export CUDA_VISIBLE_DEVICES=1
# general configureation
backend=pytorch
stage=4
stop_stage=100
ngpu=1
debugmode=1
dumpdir=dump_lhuc
N=0
verbose=0
# resume=exp/train_nodup_trim_pytorch_train_pytorch_transformer_specaug/results/snapshot.ep.2
resume=
# exp/1024/sat/train_nodup_trim_sp_satsc_conv2d2_sigmoid_train_conformer_lhuc.satsc.old_specaug/results/snapshot.ep.98
# feature configuration
do_delta=false

preprocess_config=conf/specaug.yaml
#conf/no_preprocess.yaml
#train_config=conf/1007/train_transformer_lhuc.supervised.yaml # change to transformer
train_config=conf/train_conformer_lhuc.sat.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.sat.yaml

# rnnlm related
lm_resume=
lmtag=

# decoding parameter
recog_model=model.acc.best # set a best model to be used for decoding
n_average=10
lang_model=rnnlm.model.best

# bpemode (unigram or bpe)
nbpe=2000
bpemode=bpe

# exp tag
tag_dean=satsc_conv2d2_sigmoid
tag=
# tag="transformer_baseline_lhuc_100ep_testadapt_lr10.0_batch64_accgrad1_nopreprocess_conv2d2_sigmoid_epoch10_unsupervised_noshuffle" # tag for managing experiments

. utils/parse_options.sh || exit 1;

set -e #-e 'error'
set -u #-u 'undefined variable'
set -o pipefail 'error in pipeline'

train_set=train_nodup_trim
train_dev=train_dev_trim
#train_set=eval2000
#train_dev=eval2000
recog_set="eval2000"

feat_tr_dir=dump_lhuc/train_nodup_trim/deltafalse
feat_dt_dir=dump_lhuc/train_dev_trim/deltafalse

# feat_tr_dir=dump_sp_lhuc/train_nodup_trim_sp/deltafalse
# feat_dt_dir=dump_sp_lhuc/train_dev_trim/deltafalse

dict=data/lang_char/train_nodup_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/train_nodup_${bpemode}${nbpe}

if [ -z ${tag} ]; then
    expname=${train_set}_${tag_dean}_$(basename ${train_config%.*})
    if ${do_delta}; then
	    expname=${expname}_delta
    fi
    if [ -n "${preprocess_config}" ]; then
	    expname=${expname}_$(basename ${preprocess_config%.*})
    fi
else
    expname=${train_set}_${tag}
fi
expdir=exp/1024/sat/${expname}
mkdir -p ${expdir}

speaker_num=`cat ${feat_tr_dir}/\num_spk`
# speaker_num=80
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json \
        --speaker-num ${speaker_num}
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=32

    # 
    recog_model=model.last${n_average}.avg.best
    average_checkpoints.py --backend ${backend} \
                --snapshots ${expdir}/results/snapshot.ep.* \
                --out ${expdir}/results/${recog_model} \
                --num ${n_average}
    
    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        # recog_model=snapshot.ep.10
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}    
        # eval_speakernum=`cat ${feat_tr_dir}/\num_spk`

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json
        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} 
            
			# --rnnlm ${lmexpdir}/${lang_model} \
			# --api v2

	# this is required for local/score_sclite.sh to get hyp.wrd.trn
        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
	if [[ "${decode_dir}" =~ "eval2000" ]]; then
        local/score_sclite.sh data/eval2000 ${expdir}/${decode_dir}
	elif [[ "${decode_dir}" =~ "rt03" ]]; then
	    local/score_sclite.sh data/rt03 ${expdir}/${decode_dir}
	fi
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"  
fi


