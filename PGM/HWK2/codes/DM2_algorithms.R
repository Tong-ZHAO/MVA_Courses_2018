if (!require("mvtnorm")) install.packages("mvtnorm")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("gridExtra")) install.packages("gridExtra")
if (!require("extrafont")) install.packages("extrafont")

#################################  kmeans functions ###################################

distance_to_given_point<-function(data,x,method="euclidian"){
  #### Function that computes the distance of all elements of a dataset to a given point
  ## Input
  
  # data= dataset (matrix of dimension nxp)
  # x=coordinates of a given point (vector of dimension p)
  
  ## Output
  # distance= vector with the computed distance (vector of dimension n)
  
  distance=apply(data,1,function(y) dist(rbind(y,x),method))
  return(distance)
}

kmeans_initialization<-function(data,k,seed=0,method="euclidian")
{
  #### Function that initializes the centroids usin the kmeans+ algorithm
  ## Input
  
  # data= dataset (matrix of dimension nxp)
  # k=number of centroids to be generated (integer number)
  # seed= seed to be used in a pseudorandom number generator
  # method= type of distance among the options "euclidean", "maximum", "manhattan", "canberra", "binary" or "minkowski"
  
  
  ### Output
  
  # centrois= points chosen as initial centroids (matrix of dimension kxp)
  set.seed(seed)
  centroids=data[sample(1:nrow(data),1),]
  for(i in 2:k){
    aux_distance=apply(centroids,1,function(y) distance_to_given_point(data,y,method))
    distance_nearest_centroid=apply(aux_distance,1,min)
    probability_selection=(distance_nearest_centroid^2)/sum(distance_nearest_centroid^2)
    new_centroid=data[sample(x=1:nrow(data),size=1,prob=probability_selection),]
    centroids<-rbind(centroids,new_centroid)
  }
  return(centroids)
}


kmeans_maximization<-function(data,ini_centroids,seed=0,method="euclidian"){
  ## Function that performs the maximization of the dispertion function 
  ## in the kmeans algorithm given an initial point
  
  ## Input
  # data= dataset (matrix of dimension nxp)
  # ini_centrois= points chosen as initial centroids (matrix of dimension kxp)
  # seed= seed to be used in a pseudorandom number generator
  # method= type of distance among the options "euclidean", "maximum", "manhattan", "canberra", "binary" or "minkowski"
  
  
  ### Output
  # labels= categories asigned to each of the data points
  
  centroids_old=ini_centroids
  distance_matrix<-apply(centroids_old,1,function(y) distance_to_given_point(data,y,method))
  label_old=apply(distance_matrix,1,which.min)
  centroids_new=apply(data,2,function(x) by(x,label_old,mean))
  distance_matrix<-apply(centroids_new,1,function(y) distance_to_given_point(data,y,method))
  label_new=apply(distance_matrix,1,which.min)
  
  while(sum(label_new!=label_old)>0){
    centroids_old=centroids_new
    distance_matrix<-apply(centroids_old,1,function(y) distance_to_given_point(data,y,method))
    label_old=apply(distance_matrix,1,which.min)
    centroids_new=apply(data,2,function(x) by(x,label_old,mean))
    distance_matrix<-apply(centroids_new,1,function(y) distance_to_given_point(data,y,method))
    label_new=apply(distance_matrix,1,which.min)
  }
  
  return(list(label=label_new,centroids=centroids_new))
  
}

kmeans_plus_fit<-function(data,k,seed=0,method="euclidian"){
  ## Function that performs the kmeans_plus algorithm
  
  ## Input
  # data= dataset (matrix of dimension nxp)
  # k=number of centroids to be generated (integer number)
  # seed= seed to be used in a pseudorandom number generator
  # method= type of distance among the options "euclidean", "maximum", "manhattan", "canberra", "binary" or "minkowski"
  
  ### Output
  # labels= categories asigned to each of the data points (vector of dimension n)
  # centroids= centroids estimated by the kmeans_algorithm (matrix of dimension kxp) 
  
  ini_centroids=kmeans_initialization(data,k,seed,method)
  results=kmeans_maximization(data,ini_centroids,method)
  return(list(label=results$label,centroids=results$centroids))
}

kmeans_plus_predict<-function(data,centroids,method="euclidian"){
  ## Function that predicts the label of new data poins by using the
  ## centroids found by the kmeans algorithm
  
  ## Input
  # data= dataset (matrix of dimension nxp)
  # centroids= centroids estimated by the kmeans_algorithm (matrix of dimension kxp)
  
  ### Output
  # labels= categories asigned to each of the data points (vector of dimension n)
  
  distance_matrix<-apply(centroids,1,function(y) distance_to_given_point(data,y,method))
  label=apply(distance_matrix,1,which.min)
  return(label)
}

graph_kmeans<-function(data,centroids,label,method,seed){
  ###### Graph to construct the Graphs of the report
  ## Input
  # data= dataset (matrix of dimension nxp)
  # centroids=matrix containing the centroids (matrix of dimension k*p)
  # label= class predicted for each datapoint
  # seed= seed to be used in a pseudorandom number generator
  # method= type of distance among the options "euclidean", "maximum", "manhattan", "canberra", "binary" or "minkowski"

  ## Output
  # g= ggplot object of the graph
 
  centroids=data.frame(centroids)
  coords = paste(round(centroids$x1,2),round(centroids$x2,2),sep=",")
  
  g<-ggplot(data=data)+geom_point(aes(x1,x2,col=as.factor(label)),alpha=0.5)+
    geom_label(data=centroids,aes(x1,x2,label=coords))+
    ggtitle(paste0("K-means with ",method," distance \n and seed= ",seed))+
    xlim(c(-11,11))+ylim(c(-11,11))+guides(col=FALSE)
  
  return(g)
}


######################### EM for Gaussian with sigma proportional to identity################



E_step_EM_gaussian_Id<-function(data,p,mu,variance){
  ## Function estimating p(z=j|x,tetha) 
  
  ## Input
  # data= dataset (matrix of dimension nxp)
  # mu=matrix containing the mean vectors (matrix of dimension k*p)
  # variance= variance vector (vector of dimension k)
  # p= vector containing the probabilities p(z=j) (vector of dimension k)
  
  ## output
  # tau= matrix of the estimated probabilities p(z=j|x_i,tetha) (matrix of dimension nxk)
  
  numerators=matrix(rep(0,nrow(mu)*nrow(data)),ncol=nrow(mu))
  for(i in 1:nrow(mu)){
  numerators[,i]=p[i]*dmvnorm(data,mean=mu[i,],sigma=variance[i]*diag(ncol(data))) 
  }
  denominators=apply(numerators,1,sum)
  tau=numerators/denominators
  return(tau)
}

M_step_EM_gaussian_Id<-function(data,tau)
{
  ## Function updating  parameters mu,variance,p 
  
  ## Input
  # data= dataset (matrix of dimension nxp)
  # tau= matrix of the estimated probabilities p(z=j|x_i,tetha) (matrix of dimension nxk)
  
  ## Output
  # mu=matrix containing the mean vectors (matrix of dimension k*p)
  # variance= variance vector (vector of dimension k)
  # p= vector containing the probabilities p(z=j) (vector of dimension k)
  
  mu=matrix(rep(0,ncol(data)*ncol(tau)),ncol=ncol(data))
  variance=rep(0,ncol(tau))
  p=rep(0,ncol(tau))
  for(j in 1:ncol(tau)){
    p[j]=mean(tau[,j])
    mu[j,]=apply(data,2,function(x) weighted.mean(x,tau[,j]))
    data_centered=data-mu[j,]
    aux_variance=apply(data_centered*data_centered,1,sum)
    variance[j]=weighted.mean(aux_variance,tau[,j])/2
  }
  return(list(p=p,mu=mu,variance=variance))
}

EM_gaussian_Id_fit<-function(data,k,maxiter=1000,eps=1e-5,verbose=FALSE){
  ## Function performing the EM algorithm with initilization using kmeans
  
  ##Input
  # data= dataset (matrix of dimension nxp)
  # maxiter= number of iteractions
  # eps=desired precision 
  # k=number of mixtures
  # verbose = TRUE if you one to print the loglikelihood of each iteraction
  
  ## Output
  # mu=matrix containing the mean vectors (matrix of dimension k*p)
  # variance= variance vector (vector of dimension k)
  # p= vector containing the probabilities p(z=j) (vector of dimension k)
  # tau= matrix of the estimated probabilities p(z=j|x_i,tetha) (matrix of dimension nxk)
  # loglikelihood=loglikelihood of a mixture of gaussian distributions
  
  # Initialization with K-means
  results=kmeans_plus_fit(data,k)
  mu_old=results$centroids
  label_old=results$label
  variance_old<-c()
  
  for(j in 1:nrow(mu_old)){
     data_j=data[label_old==j,]
     data_centered=data_j-mu_old[j,]
     aux_variance=apply(data_centered*data_centered,1,mean)
     aux_variance=mean(aux_variance)
     variance_old=c(variance_old,aux_variance)
   }
  p_old=table(label_old)/length(label_old)
  loglikelihood_old<-loglikelihood_mixtureGaus(data,p_old,mu_old,variance=sapply(variance_old,function(x) x*diag(ncol(mu_old)),simplify=FALSE))
  tau_old=c()
  
  ### First iteraction EM
  ### E Step
  tau_new=E_step_EM_gaussian_Id(data,p_old,mu_old,variance_old)
  ### M_Step
  M_step_res=M_step_EM_gaussian_Id(data,tau_new)
  p_new=M_step_res$p
  mu_new=M_step_res$mu
  variance_new=M_step_res$variance
  label_new=apply(tau_new,1,which.max)
  loglikelihood_new<-loglikelihood_mixtureGaus(data,p_new,mu_new,variance=sapply(variance_new,function(x) x*diag(ncol(mu_new)),simplify=FALSE))

  i=1
  while(i<maxiter && loglikelihood_new-loglikelihood_old>eps){
    ### Actualization of parameters
    p_old=p_new
    mu_old=mu_new
    variance_old=variance_new
    label_old=label_new
    loglikelihood_old=loglikelihood_new 
    tau_old=tau_new
    ### E Step
    tau_new=E_step_EM_gaussian_Id(data,p_old,mu_old,variance_old)
    ### M_Step
    M_step_res=M_step_EM_gaussian_Id(data,tau_new)
    p_new=M_step_res$p
    mu_new=M_step_res$mu
    variance_new=M_step_res$variance
    label_new=apply(tau_new,1,which.max)
    loglikelihood_new<-loglikelihood_mixtureGaus(data,p_new,mu_new,variance=sapply(variance_new,function(x) x*diag(ncol(mu_new)),simplify=FALSE))
    if(verbose){print(loglikelihood_new)}
    i=i+1
  }
  return(list(p=p_old,mu=mu_old,variance=variance_old,tau=tau_old,label=label_old,loglikelihood=loglikelihood_old))
}

EM_gaussian_Id_predict<-function(data,p,mu,variance){
  ## Function estimating p(z=j|x,tetha) and the label of each data point
  
  ## Input
  # data= dataset (matrix of dimension nxp)
  # mu=matrix containing the mean vectors (matrix of dimension k*p)
  # variance= variance vector (vector of dimension k)
  # p= vector containing the probabilities p(z=j) (vector of dimension k)
  
  ## output
  # tau= matrix of the estimated probabilities p(z=j|x_i,tetha) (matrix of dimension nxk)
  # label=labels predicted by the EM algorithm (vector of dimension n)
  
  tau=E_step_EM_gaussian_Id(data,p,mu,variance)
  label=apply(tau,1,which.max)
  return(list(label=label,tau=tau))
}

################################# EM for Gaussian mixture ##########################

E_step_EM_gaussian<-function(data,p,mu,variance){
  ## Function estimating p(z=j|x,tetha) 
  
  ## Input
  # data= dataset (matrix of dimension nxp)
  # mu=matrix containing the mean vectors (matrix of dimension k*p)
  # variance= list of covariance matrices (list of length k with covariance matrices)
  # p= vector containing the probabilities p(z=j) (vector of dimension k)
  
  ## output
  # tau= matrix of the estimated probabilities p(z=j|x_i,tetha) (matrix of dimension nxk)
  
  numerators=matrix(rep(0,nrow(mu)*nrow(data)),ncol=nrow(mu))
  for(i in 1:nrow(mu)){
    numerators[,i]=p[i]*dmvnorm(data,mean=mu[i,],sigma=variance[[i]]) 
  }
  denominators=apply(numerators,1,sum)
  tau=numerators/denominators
  return(tau)
}

M_step_EM_gaussian<-function(data,tau)
{
  ## Function updating  parameters mu,variance,p 
  
  ## Input
  # data= dataset (matrix of dimension nxp)
  # tau= matrix of the estimated probabilities p(z=j|x_i,tetha) (matrix of dimension nxk)
  
  ## Output
  # mu=matrix containing the mean vectors (matrix of dimension k*p)
  # variance= list of covariance matrices (list of length k with covariance matrices)
  # p= vector containing the probabilities p(z=j) (vector of dimension k)
  
  mu=matrix(rep(0,ncol(data)*ncol(tau)),ncol=ncol(data))
  variance=list()
  p=rep(0,ncol(tau))
  for(j in 1:ncol(tau)){
    p[j]=mean(tau[,j])
    mu[j,]=apply(data,2,function(x) weighted.mean(x,tau[,j]))
    variance[[j]]=cov.wt(data,tau[,j],method="ML",center=mu[j,])$cov
 }
  return(list(p=p,mu=mu,variance=variance))
}

EM_gaussian_fit<-function(data,k,maxiter=100,eps=1e-5,verbose=FALSE){
  ## Function performing the EM algorithm with initilization using kmeans
  
  ##Input
  # data= dataset (matrix of dimension nxp)
  # maxiter= maximum number of iteractions
  # eps= desired precision in the loglikelihood
  # k=number of mixtures
  # verbose = TRUE if you one to print the loglikelihood of each iteraction
  
  ## Output
  # mu=matrix containing the mean vectors (matrix of dimension k*p)
  # variance= list of covariance matrices (list of length k with covariance matrices)
  # p= vector containing the probabilities p(z=j) (vector of dimension k)
  # tau= matrix of the estimated probabilities p(z=j|x_i,tetha) (matrix of dimension nxk)
  # loglikelihood=loglikelihood of a mixture of gaussian distributions
  
  # Initialization
  results=kmeans_plus_fit(data,k)
  mu_old=results$centroids
  label_old=results$label
  variance_old=by(data,label_old,cov,simplify = FALSE)
  p_old=table(label_old)/length(label_old)
  loglikelihood_old<-loglikelihood_mixtureGaus(data,p_old,mu_old,variance=variance_old)
  tau_old<-c()
  
  ### First iteraction EM
  ### E Step
  tau_new=E_step_EM_gaussian(data,p_old,mu_old,variance_old)
  ### M_Step
  M_step_res=M_step_EM_gaussian(data,tau_new)
  p_new=M_step_res$p
  mu_new=M_step_res$mu
  variance_new=M_step_res$variance
  label_new=apply(tau_new,1,which.max)
  loglikelihood_new<-loglikelihood_mixtureGaus(data,p_new,mu_new,variance=variance_new)
  
  
  # E and M step
  i=1
  while(i<maxiter && loglikelihood_new-loglikelihood_old>eps){
    
    ### Actualization of parameters
    p_old=p_new
    mu_old=mu_new
    variance_old=variance_new
    label_old=label_new
    loglikelihood_old=loglikelihood_new 
    tau_old=tau_new
    ### E Step
    tau_new=E_step_EM_gaussian(data,p_old,mu_old,variance_old)
    ### M Step
    M_step_res=M_step_EM_gaussian(data,tau_new)
    p_new=M_step_res$p
    mu_new=M_step_res$mu
    variance_new=M_step_res$variance
    label_new=apply(tau_new,1,which.max)
    loglikelihood_new=loglikelihood_mixtureGaus(data,p_new,mu_new,variance_new)
    if(verbose){print(loglikelihood_new)}
    
    i=i+1
  }
  return(list(p=p_old,mu=mu_old,variance=variance_old,tau=tau_old,label=label_old,loglikelihood=loglikelihood_old))
}

EM_gaussian_predict<-function(data,p,mu,variance){
  ## Function estimating p(z=j|x,tetha) and the label of each data point
  
  ## Input
  # data= dataset (matrix of dimension nxp)
  # mu=matrix containing the mean vectors (matrix of dimension k*p)
  # variance= list of covariance matrices (list of length k with covariance matrices)
  # p= vector containing the probabilities p(z=j) (vector of dimension k)
  
  ## output
  # tau= matrix of the estimated probabilities p(z=j|x_i,tetha) (matrix of dimension nxk)
  # label=labels predicted by the EM algorithm (vector of dimension n)
  
  tau=E_step_EM_gaussian(data,p,mu,variance)
  label=apply(tau,1,which.max)
  return(list(label=label,tau=tau))
}


################################# other functions ################################

loglikelihood_mixtureGaus<-function(data,p,mu,variance){
  ## It estimates the looklihood of a mixture of Gaussians
  
  ## Input
  # data= dataset (matrix of dimension nxp)
  # mu=matrix containing the mean vectors (matrix of dimension k*p)
  # variance= list of covariance matrices (list of length k with covariance matrices)
  # p= vector containing the probabilities p(z=j) (vector of dimension k)
  
  ## Output
  # loglikelihood=loglikelihood of a mixture of gaussian distributions
  
   aux_loglikelihood=matrix(rep(0,nrow(mu)*nrow(data)),ncol=nrow(mu))
   for(i in 1:nrow(mu)){
     aux_loglikelihood[,i]=p[i]*dmvnorm(data,mean=mu[i,],sigma=variance[[i]]) 
   }
  
   loglikelihood_i<-apply(aux_loglikelihood,1,function(x) log(sum(x)))
   loglikelihood=sum(loglikelihood_i)
  return(loglikelihood) 
}

ellipse_equation<-function(n_points,confidence=0.95,mu,variance){
  ## It estimates the coordinates of a confidence ellipse of a multivariate
  ## gaussian distribution given its vector mean and covariance matrix
  
  # Input
  # n_points=number of coordinates to stimate (integer>0)
  # confidence= confidence threshold (float between 0 and 1)
  # mu= vector of means(vector of dimension k)
  # variance = covariance matrix (matrix of dimension kxk)
  
  # Output 
  # coordinates= dataframe with the coordinates of the ellipse
  
   quantiles<-qchisq(confidence,2)
   eigen_descomp<-eigen(variance)
   
   axis_length=2*sqrt(quantiles*eigen_descomp$values)
   orientation_vector=eigen_descomp$vectors[,which.max(c(variance[2,2],variance[1,1]))]
   orientation=acos((c(1,0)%*%orientation_vector))
   tetha=seq(0,2*pi,length.out=n_points)
   x=(axis_length[1]/2)*cos(tetha)
   y=(axis_length[2]/2)*sin(tetha)
   if(eigen_descomp$values[2]==eigen_descomp$values[1]){
     x=x+mu[1]
     y=y+mu[2]
     coordinates=data.frame(x=x,y=y)
     return(coordinates)
   }else{
   x_rotated=x*c(cos(orientation))-y*c(sin(orientation))+mu[1]
   y_rotated=x*c(sin(orientation))+y*c(cos(orientation))+mu[2]
   coordinates=data.frame(x=x_rotated,y=y_rotated)
   }
   return(coordinates)
}

graph_mixtureGaus<-function(data,p,mu,variance,label){
  ## Input
  # data= dataset (matrix of dimension nxp)
  # mu=matrix containing the mean vectors (matrix of dimension k*p)
  # variance= list of covariance matrices (list of length k with covariance matrices)
  # p= vector containing the probabilities p(z=j) (vector of dimension k)
  # label= class predicted for each datapoint
  
  ## Output
  # g= ggplot object of the graph
  color=c("red","black","blue","green")
  ellipses=mapply(function(mu,variance) ellipse_equation(100,0.90,mu,variance),
                  as.list(data.frame(t(mu))),variance,SIMPLIFY = FALSE)
  ellipses=Reduce(rbind,ellipses)
  ellipses=data.frame(ellipses)
  ellipses$label=as.factor(matrix(sapply(1:4,function(x) as.character(rep(x,100))),ncol=1))
  data$label=as.factor(as.character(label))
  centroids=data.frame(mu)
  coords = paste(round(centroids$X1,2),round(centroids$X2,2),sep=",")
  g<-ggplot(data=data)+geom_point(aes(x1,x2,col=data$label))+theme(legend.position = "none")+
    geom_label(data=centroids,aes(X1,X2,label=coords),label.size = 0.10)+
    geom_point(data=as.data.frame(mu),aes(V1,V2,col=as.factor(1:4)),shape=3,size=3)+
    geom_polygon(data=ellipses,aes(x=x,y=y,fill=label,col=label),alpha=0.2)+
    xlim(c(-11,11))+ylim(c(-12,12))+guides(col=FALSE)
    
  return(g)
}



