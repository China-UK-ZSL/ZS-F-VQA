data=3;
ke=10;
kr=3;
score=10;
zsl=0;
python joint_test.py --gpu_id 1 --exp_name fusion_prediction --ZSL "${zsl}" --exp_id rel"${kr}"_fact"${ke}"data_"${data}"score_"${score}" --data_choice "${data}" --top_rel "${kr}" --top_fact "${ke}" --soft_score "${score}"  --mrr 1