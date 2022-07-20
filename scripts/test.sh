################################caption###############################
task=caption
mask_num=1
device=4
bs=16

#image

cn=img
mask_name=img
lr1=1e-5
lr2=1e-5

python main.py --cuda ${device} \
--mode 1 --model_file_name epoch3_f1_0.87504.pth \
--auxiliary_task_split_file datasets/${task}/ \
--auxiliary_task_img_dir datasets/${task}/img/ \
--task_name ${task} \
--model_dir save_model/${task}_mask_${mask_name}${mask_num}_${lr1}_${lr2}/ \
--class_name ${cn} --mask_part ${mask_name}_${mask_num} \
--dropout 0.5 --batch_size ${bs} \
--lr_1 ${lr1} --lr_2 ${lr2}

#text

cn=txt
mask_name=txt
lr1=1e-5
lr2=1e-4

python main.py --cuda ${device} \
--mode 1 --model_file_name epoch4_f1_0.56981.pth \
--auxiliary_task_split_file datasets/${task}/ \
--auxiliary_task_img_dir datasets/${task}/img/ \
--task_name ${task} \
--model_dir save_model/${task}_mask_${mask_name}${mask_num}_${lr1}_${lr2}/ \
--class_name ${cn} --mask_part ${mask_name}_${mask_num} \
--dropout 0.5 --batch_size ${bs} \
--lr_1 ${lr1} --lr_2 ${lr2}


##image+txt

cn=it
mask_name=img
lr1=2e-5
lr2=1e-4

python main.py --cuda ${device} \
--mode 1 --model_file_name epoch10_f1_0.51238.pth \
--auxiliary_task_split_file datasets/${task}/ \
--auxiliary_task_img_dir datasets/${task}/img/ \
--task_name ${task} \
--model_dir save_model/${task}_mask_${mask_name}${mask_num}_${lr1}_${lr2}/ \
--class_name ${cn} --mask_part ${mask_name}_${mask_num} \
--dropout 0.5 --batch_size ${bs} \
--lr_1 ${lr1} --lr_2 ${lr2}

#################################senti###############################
task=senti
mask_num=0
bs=16

##image

cn=img
mask_name=img
lr1=1e-5
lr2=1e-5

python main.py --cuda ${device} \
--mode 1 --model_file_name epoch2_f1_0.86970.pth \
--auxiliary_task_split_file datasets/${task}/ \
--auxiliary_task_img_dir datasets/${task}/img/ \
--task_name ${task} \
--model_dir save_model/${task}_mask_${mask_name}${mask_num}_${lr1}_${lr2}/ \
--class_name ${cn} --mask_part ${mask_name}_${mask_num} \
--dropout 0.5 --batch_size ${bs} \
--lr_1 ${lr1} --lr_2 ${lr2}

#text

cn=txt
mask_name=img
lr1=1e-5
lr2=1e-4

python main.py --cuda ${device} \
--mode 1 --model_file_name epoch4_f1_0.57494.pth \
--auxiliary_task_split_file datasets/${task}/ \
--auxiliary_task_img_dir datasets/${task}/img/ \
--task_name ${task} \
--model_dir save_model/${task}_mask_${mask_name}${mask_num}_${lr1}_${lr2}/ \
--class_name ${cn} --mask_part ${mask_name}_${mask_num} \
--dropout 0.5 --batch_size ${bs} \
--lr_1 ${lr1} --lr_2 ${lr2}

##image+txt

cn=it
mask_name=img
lr1=1e-5
lr2=1e-4

python main.py --cuda ${device} \
--mode 1 --model_file_name epoch7_f1_0.47029.pth \
--auxiliary_task_split_file datasets/${task}/ \
--auxiliary_task_img_dir datasets/${task}/img/ \
--task_name ${task} \
--model_dir save_model/${task}_mask_${mask_name}${mask_num}_${lr1}_${lr2}/ \
--class_name ${cn} --mask_part ${mask_name}_${mask_num} \
--dropout 0.5 --batch_size ${bs} \
--lr_1 ${lr1} --lr_2 ${lr2}



#################################fakenews###############################

task=fake_news
mask_num=0
bs=16

# ##image

cn=img
mask_name=txt
lr1=1e-5
lr2=1e-5
# epoch2_f1_0.87533.pth
python main.py --cuda ${device} \
--mode 1 --model_file_name epoch2_f1_0.87533.pth \
--auxiliary_task_split_file datasets/${task}/ \
--auxiliary_task_img_dir datasets/${task}/img/ \
--task_name ${task} \
--model_dir save_model/${task}_mask_${mask_name}${mask_num}_${lr1}_${lr2}/ \
--class_name ${cn} --mask_part ${mask_name}_${mask_num} \
--dropout 0.5 --batch_size ${bs} \
--lr_1 ${lr1} --lr_2 ${lr2}

##text

cn=txt
mask_name=img
lr1=1e-5
lr2=1e-4

python main.py --cuda ${device} \
--mode 1 --model_file_name epoch4_f1_0.56388.pth \
--auxiliary_task_split_file datasets/${task}/ \
--auxiliary_task_img_dir datasets/${task}/img/ \
--task_name ${task} \
--model_dir save_model/${task}_mask_${mask_name}${mask_num}_${lr1}_${lr2}/ \
--class_name ${cn} --mask_part ${mask_name}_${mask_num} \
--dropout 0.5 --batch_size ${bs} \
--lr_1 ${lr1} --lr_2 ${lr2}

##image+txt

cn=it
mask_name=txt
lr1=1e-5
lr2=3e-4

python main.py --cuda ${device} \
--mode 1 --model_file_name epoch25_f1_0.47938.pth \
--auxiliary_task_split_file datasets/${task}/ \
--auxiliary_task_img_dir datasets/${task}/img/ \
--task_name ${task} \
--model_dir save_model/${task}_mask_${mask_name}${mask_num}_${lr1}_${lr2}/ \
--class_name ${cn} --mask_part ${mask_name}_${mask_num} \
--dropout 0.5 --batch_size ${bs} \
--lr_1 ${lr1} --lr_2 ${lr2}