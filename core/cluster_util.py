import tensorflow as tf

def target_distribution(batch:tf.Tensor)->tf.Tensor:
    """
    return the target distribution of data
    
    Args:
        batch:2-D Tensor
    Returns:
        target:2-D Tensor
    """
    squared=tf.square(batch,name="square op")
    reduce_sum=tf.reduce_sum(batch,axis=0)
    target=tf.divide(squared,reduce_sum)
    return target


def KLDiv(predict:tf.Tensor,target:tf.Tensor,eps=1e-08)->float:
    p1=predict+eps
    t1=target+eps
    logP=tf.math.log(p1)
    logT=tf.math.log(t1)
    log_sub=logP-logT
    kld=tf.multiply(target,log_sub)
    kld=tf.reduce_sum(kld,axis=1)
    kld=tf.reduce_mean(kld)
    return kld
    
    