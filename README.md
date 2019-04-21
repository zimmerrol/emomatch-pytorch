# EmoMatch Task
Simple transfer-learning task based on the VoxCeleb dataset to pretrain networks working on videos (audio + video)
This code requires you to download the VoxCeleb dataset and to extract it (both audio and video).

The idea of this aproach is based on the paper [Look, Listen, Learn](https://arxiv.org/abs/1705.08168): here, audio and video information were used to pretain an image encoder network to be used for image classificaiton tasks.

This project tries to extend this approach to not only train an image encoder but to actually pre-train a network that is able to process both audio and video information. The task the network is meant to solve is rather simple: given an audio sequence and a video sequence, decide whether the two match (i.e. have the same origin).


Structure of the EmoMatch training procedure is shown in the image below. 
![](avnet_model.png)
The left side showsthe data preparation while the right side illustrates the data flow through thenetwork.  In  the  data  preparation  video  recordings  are  used  to  separate  theirvideo and audio track. These tracks are then feed into a VNet and an ANet for the video respectively the audio. These networks serve as an encoder to generate features for a classifier network. This classifier will then detect whether the audiotrack originates from the same original recording as the video track (Match) or from a two different recordings (No Match).
