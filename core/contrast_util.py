import tensorflow as tf

def contrasive_loss(features:tf.Tensor,base_temprature:float,temprature:float):
    if len(features.shape)!=3:
        raise ValueError("we expected dims with 3,but got {}".format(len(features.shape)))
    batch_size=features.shape[0]
    mask=tf.eye(batch_size,dtype=tf.float32)
    
    contrast_count=features.shape[1]
    
    contrast_features=tf.split(features,contrast_count,axis=1)
    contrast_features_cat=tf.concat(contrast_features,axis=0) #concat these tensors
    contrast_features_cat=tf.squeeze(contrast_features_cat,axis=1) #delete the axis=1
    
    contrast_features_trans=tf.transpose(contrast_features_cat)
    #compute the distance,the vector should mod 1
    anchor_dot_contrast=tf.divide(
        tf.matmul(contrast_features_cat,contrast_features_trans),temprature
    )

    #for numerical stability
    logits_max=tf.reduce_max(anchor_dot_contrast,axis=1,keepdims=True)
    logits=anchor_dot_contrast-logits_max
    # a foo method
    mask_repeat=tf.concat(
        [mask,mask],axis=1
    )
    mask_repeat=tf.concat(
        [mask_repeat,mask_repeat],axis=0
    )

    size=batch_size*contrast_count
    mask_diag=tf.eye(size)
    mask_ones=tf.ones(size)
    logits_mask=mask_ones-mask_diag
    
    mask_pos=mask_repeat*logits_mask
    
    #compute log_prob
    exp_logits=tf.exp(logits)*logits_mask #exclude self distance 
    exp_logits_sum=tf.reduce_sum(exp_logits,axis=1,keepdims=True) #compute the distance sum exlude self 
    log_prob=logits-tf.math.log(exp_logits_sum) #按照公式来说是 log(exp(logits))-log(sum)=logits-log(sum)

    #compute the mean of log over positive
    log_prob_pos=(mask_pos*log_prob)
    mean_log_prob_pos=tf.reduce_sum(log_prob_pos,axis=1)/tf.reduce_sum(mask_pos,axis=1)

    #loss 
    loss=-(temprature/base_temprature)*mean_log_prob_pos
    loss=tf.reduce_mean(loss)
    return loss
    


    