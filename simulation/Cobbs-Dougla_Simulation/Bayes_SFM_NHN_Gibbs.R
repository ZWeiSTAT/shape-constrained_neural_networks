library(truncnorm)
Bayeslinear_NHN_fit<-function(B_burnin=5000,
                              B_afterburnin=15000,
                              simdt,beta0_init = beta0,
                              beta_init= rep(1,length=dim(simdt)[2]-1),
                              sigu_init=sigmau,
                              sigv_init=sigmav,
                              df_v = 5,
                              S_v = (5+2)*sigmav^2,
                              df_u = 5,
                              S_u = (5+2)*sigmau^2
){
#simdt has k+1 columns first column is the response variable lny
#last k columns are the independent variables lnx1,\ldots,lnx_{k}
#####################################
###   Gibbs Sampler #################
#####################################
  # B_burnin=50
  # B_afterburnin=150
B=  B_burnin + B_afterburnin
#####################################
#set initial values
#####################################
n<-dim(simdt)[1]
K<-dim(simdt)[2]-1
Y<-simdt[,1]
Xdes<-as.matrix(simdt[,-1])


u_init = rep(.5,n)



#####################################
#set hyperparameter values
#####################################
sigb0_hyper=sqrt(10^6)
mub0_hyper=0#beta0
sigbeta_hyper=sqrt(10^6)
mubeta_hyper=rep(0,K)#c(beta1,beta2)

#####################################
# Store posterior samples
#####################################
sigv_post<-c(sigv_init)
sigu_post<-c(sigu_init)

beta0_post<-c(beta0_init)
beta_post<-matrix(beta_init,nrow = 1,ncol = dim(Xdes)[2])
Uvec_post<-matrix(u_init,nrow = 1,ncol = dim(Xdes)[1])
for(b in 1:B){
  if(b%%1000==0){
    cat("Cumulative average: iter=",b,"beta0=",round(mean(beta0_post[1:b]),4),"beta=",
        round(apply(as.matrix(beta_post[1:b,]),2,mean),4),"sigmav=",
        round(mean(sigv_post[1:b]),4),"sigmau=",
        round(mean(sigu_post[1:b]),4),"\n")
  }

  #####################################
  # current parameter values
  #####################################
  beta_current <- beta_post[b,]
  Uvec_current <- Uvec_post[b,]
  sig_u_current <- sigu_post[b]
  sig_v_current <- sigv_post[b]
  beta0_current <- beta0_post[b]
  beta_current <- beta_post[b,]

  #####################################
  # Update \hat{beta}_0
  #####################################
  # sigb0_hyper=sqrt(1)
  # mub0_hyper=beta0
  mu_i_star = Y - Xdes%*%beta_current + Uvec_current
  beta0_current = rnorm(1, mean = (sigb0_hyper^2*sum(mu_i_star)+sig_v_current^2*mub0_hyper)/(sig_v_current^2+n*sigb0_hyper^2),
                        sd = sqrt(sigb0_hyper^2*sig_v_current^2/(sig_v_current^2+n*sigb0_hyper^2)))
  beta0_post<-c(beta0_post,beta0_current)
  ###########################################
  # Update \hat{beta}_\ell, \ell = 1, ..., k
  ###########################################
  # sigbeta_hyper=sqrt(1)
  # mubeta_hyper=c(beta1,beta2)
  sigbeta_current<-sigbeta_hyper
  for(ell in 1:K){
    mu_ell <- Y - beta0_current- as.matrix(Xdes[,-ell])%*%as.matrix(beta_current[-ell]) + Uvec_current
    beta_current[ell]<-rnorm(1,
                             mean = (sigbeta_current^2*(sum(Xdes[,ell]*mu_ell))+sig_v_current^2*mubeta_hyper[ell])/(sig_v_current^2+sigbeta_current^2*sum(Xdes[,ell]^2)),
                             sd = sqrt((sig_v_current^2*sigbeta_current^2)/(sig_v_current^2+sigbeta_current^2*sum(Xdes[,ell]^2)))
    )
  }
  beta_post=rbind(beta_post,beta_current)
  ###########################################
  # Update Inefficiency terms U_i, i=1,...n:
  ###########################################
  #####################################
  #generate Uvec
  #####################################

  Uvec_current<-rtruncnorm(1, a=0, b=Inf,
                           mean = -sig_u_current^2*
                             (Y-(beta0_current+as.matrix(Xdes)%*%as.matrix(beta_current)))/
                             (sig_u_current^2+sig_v_current^2),
                           sd = sqrt((sig_u_current^2*sig_v_current^2)/(sig_u_current^2+sig_v_current^2))
  )

  Uvec_post<-rbind(Uvec_post,Uvec_current)

  #####################################
  #generate sigma_v^2
  #####################################
  escala_v=sum((Y-(beta0_current+as.matrix(Xdes)%*%as.matrix(beta_current))+Uvec_current)^2)+S_v
  sig_v_current<-sqrt(
    escala_v/rchisq(1,df=df_v+n)
  )

  sigv_post<-c(sigv_post,sig_v_current)

  #####################################
  #generate sigma_u
  #####################################
  escala_u=sum(Uvec_current^2)+S_u
  sig_u_current<-sqrt(
    escala_u/rchisq(1,df=df_u+n)
  )

  sigu_post<-c(sigu_post,sig_u_current)

}


est<-data.frame(beta0_post=beta0_post,beta_post=beta_post,sigv_post=sigv_post,sigu_post=sigu_post)

return(list(res=est,Upost=Uvec_post))
}

#########################
#####   Example  ########
#########################
# simulation parameters #
#########################
# beta0=0.9
# beta1=.6
# beta2=1
# sigmav=1
# sigmau=.5
# sqrt(sigmav^2/sigmau^2)
# nsamp=200
# simdt<-datagen_ind(beta0 = beta0,beta1 = beta1,beta2 = beta2,sigmav = sigmav,sigmau = sigmau,nsamp = nsamp)
#########################
# res_Bayelin<-Bayeslinear_NHN_fit(B_burnin=5000,
#                     B_afterburnin=15000,
#                     simdt,beta0_init = beta0,
#                     beta_init= c(beta1,beta2),
#                     sigu_init=sigmau,
#                     sigv_init=sigmav,
#                     df_v = 5,
#                     S_v = (5+2)*sigmav^2,
#                     df_u = 5,
#                     S_u = (5+2)*sigmau^2)

