##### This code plays the role of a main. It calls all the functions and
##### it gives a PDF with the output you asked.
##### To see the details of the implementation, check the DM1_algorithms.R file

rm(list=ls())

###### Change directories to make everything work

codes_location<-"C:/Users/Fraintendetore/Dropbox/M2MVA/Probabilistic Graphical Models/DM2/"
data_location<-"C:/Users/Fraintendetore/Dropbox/M2MVA/Probabilistic Graphical Models/DM2/classification_data_HWK2/"

###### Call algorithms file

source(paste0(codes_location,"DM2_algorithms.R"))

###### data set

EMGaus_train<-read.table(paste0(data_location,"EMGaussian.data"))
EMGaus_test<-read.table(paste0(data_location,"EMGaussian.test"))
colnames(EMGaus_train)<-colnames(EMGaus_test)<-c("x1","x2")

#####################Kmeans experiments

####### Euclidean distance with different random starts
kmeans_fit_EMGaus_euclidean=kmeans_plus_fit(EMGaus_train,k=4,seed=0,method = "euclidean")
graph_kmeans_euclidean_0<-graph_kmeans(data=EMGaus_train,centroids=kmeans_fit_EMGaus_euclidean$centroids,
             label=kmeans_fit_EMGaus_euclidean$label,method="euclidean",seed=0)
graph_kmeans_euclidean_0

kmeans_fit_EMGaus_euclidean_100=kmeans_plus_fit(EMGaus_train,k=4,seed=100,method = "euclidean")
graph_kmeans_euclidean_100<-graph_kmeans(data=EMGaus_train,centroids=kmeans_fit_EMGaus_euclidean_100$centroids,
             label=kmeans_fit_EMGaus_euclidean_100$label,method="euclidean",seed=100)
graph_kmeans_euclidean_100
### It can be seen how results change slightly
table(kmeans_fit_EMGaus_euclidean$label,kmeans_fit_EMGaus_euclidean_100$label)

######## L_inf distance
kmeans_fit_EMGaus_maximum=kmeans_plus_fit(EMGaus_train,k=4,seed=0,method = "maximum")
graph_kmeans_Linf_0=graph_kmeans(data=EMGaus_train,centroids=kmeans_fit_EMGaus_maximum$centroids,
             label=kmeans_fit_EMGaus_maximum$label,method="maximum",seed=0)
graph_kmeans_Linf_0
### Again, the results do not change too much but the algorithm converged faster
table(kmeans_fit_EMGaus_euclidean$label,kmeans_fit_EMGaus_maximum$label)

######## L_1 distance

kmeans_fit_EMGaus_manhattan=kmeans_plus_fit(EMGaus_train,k=4,seed=0,method = "manhattan")

graph_kmeans(data=EMGaus_train,centroids=kmeans_fit_EMGaus_manhattan$centroids,
             label=kmeans_fit_EMGaus_manhattan$label,method="manhattan",seed=0)
### The result don't change too much but the algorithm converged faster
table(kmeans_fit_EMGaus_euclidean$label,kmeans_fit_EMGaus_manhattan$label)


### EM algorithm with covariance proportional to identity

EM_gaussian_Id_fit_EMGaus=EM_gaussian_Id_fit(EMGaus_train,k=4,maxiter=1000,verbose=F,eps=1e-6)
variance_Id=sapply(EM_gaussian_Id_fit_EMGaus$variance,function(x) x*diag(2),simplify = F)
graph_GaussianId=graph_mixtureGaus(data=EMGaus_train,p=EM_gaussian_Id_fit_EMGaus$p,mu=EM_gaussian_Id_fit_EMGaus$mu,
                  variance=variance_Id,label=EM_gaussian_Id_fit_EMGaus$label)+ggtitle("EM algorithm for Mixture \n of Gaussians with sigma=c*I")
loglikelihood_GaussianId_train=loglikelihood_mixtureGaus(data=EMGaus_train,p=EM_gaussian_Id_fit_EMGaus$p,mu=EM_gaussian_Id_fit_EMGaus$mu,
                                                       variance=variance_Id)
loglikelihood_GaussianId_test=loglikelihood_mixtureGaus(data=EMGaus_test,p=EM_gaussian_Id_fit_EMGaus$p,mu=EM_gaussian_Id_fit_EMGaus$mu,
                                                         variance=variance_Id)
### EM algorithm for a general Gaussian Mixture

EM_gaussian_fit_EMGaus=EM_gaussian_fit(EMGaus_train,k=4,maxiter=100,verbose=T,eps=1e-6)
graph_Gaussian=graph_mixtureGaus(data=EMGaus_train,p=EM_gaussian_fit_EMGaus$p,mu=EM_gaussian_fit_EMGaus$mu,
                                   variance=EM_gaussian_fit_EMGaus$variance,label=EM_gaussian_fit_EMGaus$label)+ggtitle("EM algorithm for \n Mixture of Gaussians")
loglikelihood_Gaussian_train=loglikelihood_mixtureGaus(data=EMGaus_train,p=EM_gaussian_fit_EMGaus$p
                                                       ,mu=EM_gaussian_fit_EMGaus$mu,
                                                       variance=EM_gaussian_fit_EMGaus$variance)
loglikelihood_Gaussian_test=loglikelihood_mixtureGaus(data=EMGaus_test,p=EM_gaussian_fit_EMGaus$p
                                                      ,mu=EM_gaussian_fit_EMGaus$mu,
                                                      variance=EM_gaussian_fit_EMGaus$variance)


### Table with loglikehoods

loglikelihood_train<-c(loglikelihood_GaussianId_train,loglikelihood_Gaussian_train)
loglikelihood_test<-c(loglikelihood_GaussianId_test,loglikelihood_Gaussian_test)
  
loglikelihood<-data.frame(train=round(loglikelihood_train,2),test=round(loglikelihood_test,2))
row.names(loglikelihood)<-c("Mixture of Gaussians \n with sigma=c*I","Mixture of Gaussians")
graph_table<-tableGrob(loglikelihood)


loadfonts()
pdf(paste0(codes_location,"/figures_DM2.pdf"))

grid.arrange(arrangeGrob(graph_kmeans_euclidean_0,graph_kmeans_euclidean_100,nrow=1,ncol=2),
             arrangeGrob(graph_kmeans_Linf_0,graph_GaussianId,nrow=1,ncol=2),
             arrangeGrob(graph_Gaussian,graph_table,nrow=1,ncol=2),nrow=3)
dev.off()


