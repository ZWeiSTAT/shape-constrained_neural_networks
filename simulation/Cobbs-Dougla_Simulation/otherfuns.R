skew_rho<-function(rhoval){sqrt(2/pi)*rhoval^3*(4/pi-1)*(1-2*rhoval^2/pi)^(-3/2)}
sig_u_fun<-function(sig_dp,lam_dp){
  sig_u=sqrt(sig_dp^2*lam_dp^2/(1+lam_dp^2))
  return(sig_u)
}

sig_v_fun<-function(sig_dp,lam_dp){
  sig_v=sqrt(sig_dp^2/(1+lam_dp^2))
  return(sig_v)
}

#################
#Estimates of TE#
#################
TE_fun<-function(res,sigv,sigu){
  sig=sqrt(sigv^2+sigu^2)
  lam=sqrt(sigu^2/sigv^2)
  mu_star=-res*sigu^2/sig^2
  sig_star=sqrt(sigu^2*sigv^2/sig^2)
  tem=mu_star+sig_star*dnorm(mu_star/sig_star)/pnorm(mu_star/sig_star)
  return(exp(-tem))
}
#return mean score for each beta, sigma_u and sigma_v
TE_bayes_fun<-function(bayelinres,y,x){

  beta_post<-bayelinres[1:(length(bayelinres)-2)]
  sigv_post<-bayelinres[(length(bayelinres)-1)]
  sigu_post<-bayelinres[(length(bayelinres))]

  res=y-cbind(1,x)%*%beta_post
  sig=sqrt(sigv_post^2+sigu_post^2)
  lam=sqrt(sigu_post^2/sigv_post^2)
  mu_star=-res*sigu_post^2/sig^2
  sig_star=sqrt(sigu_post^2*sigv_post^2/sig^2)
  tem=mu_star+sig_star*dnorm(mu_star/sig_star)/pnorm(mu_star/sig_star)
  return(mean(exp(-tem)))
}

TE_bart_fun<-function(yhat_post,sigv_post,sigu_post,y_bart){
  res=y_bart-yhat_post
  sig=sqrt(sigv_post^2+sigu_post^2)
  lam=sqrt(sigu_post^2/sigv_post^2)
  mu_star=-res*sigu_post^2/sig^2
  sig_star=sqrt(sigu_post^2*sigv_post^2/sig^2)
  tem=mu_star+sig_star*dnorm(mu_star/sig_star)/pnorm(mu_star/sig_star)
  return(mean(exp(-tem)))
}


