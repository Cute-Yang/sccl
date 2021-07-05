import keras
import tensorflow as tf
from .contrast_util import contrasive_loss
from .cluster_util import (
    target_distribution,
    KLDiv
)

class SCCL:
    def __init__(self,emb_size:int,init_cluster_centers,\
        instance_out:int,alpha=1.0,base_temprature=0.7,temprature=0.7,
        rate=10,lr=1e-4,**kwargs):
        self.emb_size=emb_size
        self.instance_out=instance_out
        self.alpha=alpha
        self.base_temprature=base_temprature
        self.temprature=temprature
        self.rate=rate

        #instance head
        self.instance_model=keras.models.Sequential(
            tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(units=self.emb_size,input_shpae=(self.emb_size,)),
                    tf.keras.layers.ReLU(negative_slope=0.01), #a change
                    tf.keras.layers.Dense(units=instance_out) #linear output
                ]
            )
        )

        #cluster head
        self.cluster_centers=tf.Variable(
            value=init_cluster_centers,
            name="cluster layer",
            trainable=True #can be backward
        )
        self.optimizer=tf.optimizers.Adam(self.lr)


    def get_cluster_prob(self,embeddings):
        """
        Args:
            embeddings:tensor of ori text

        Return:
            prob:Tensor
        """ 
        embeddings=tf.expand_dims(embeddings,axis=1)
        norm_squared=tf.sum(
            (embeddings-self.cluster_centers)**2,
            axis=2
        )
        numerator=1.0/(1.0+(norm_squared/self.alpha))
        power=float(self.alpha+1)/2
        numerator=tf.pow(numerator,power)
        numerator=numerator/tf.sum(numerator,axis=1,keepdims=True) #return a 2-D Tensor
        return numerator
        
    def train_step(self,train_x0,train_x1,train_x2):
        feat1=self.instance_model(train_x1)
        feat2=self.instance_model(train_x2)
        
        feat1_mod=tf.sqrt(
            tf.sum(
                tf.square(feat1),axis=1,keepdims=True
            )
        )
        feat1=tf.divide(feat1,feat1_mod+1e-8)
        feat1_expand=tf.expand_dims(feat1,axis=1,name="expand feat1")

        feat2_mod=tf.sqrt(
            tf.sum(
                tf.square(feat2),axis=1,keepdims=True
            )
        )
        feat2=tf.divide(feat2,feat2_mod+1e-8)
        feat2=tf.expand_dims(feat2,axis=1,name="expand feat2")
        instance_logits=tf.concat(
            [feat1,feat2],
            axis=0
        )

        #conntrast loss 
        instance_loss=contrasive_loss(
            features=instance_logits,
            base_temprature=self.base_temprature,
            temprature=self.temprature
        )

        #cluster loss
        cluster_output=self.get_cluster_prob(train_x0)
        target=target_distribution(cluster_output)
        cluster_loss=KLDiv(
            predict=cluster_output,
            target=target
        )

        total_loss=instance_loss+self.rate*cluster_loss
        with tf.GradientTape() as tape:
            trainable_variables=self.instance_model.trainable_variables
            trainable_variables.extend(self.cluster_centers)
            gradients=tape.gradient(
                target=total_loss,
                sources=trainable_variables
            )
            self.optimizer.apply_gradients(
                zip(gradients,trainable_variables)
            )
            return total_loss