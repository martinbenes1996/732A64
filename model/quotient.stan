
functions {

  real[] sir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    // states
    real S = y[1];
    real E = y[2];
    real I = y[3];
    real N = x_i[1];
    //real DAYS = x_i[1];
    // parameters
    real a;
    real c;
    real b;
    real d;
    // differences
    real dS_dt;
    real dE_dt;
    real dI_dt;
    real dR_dt;
    real dD_dt;
    // time to int
    //int ti = 1;
    //while(ti < t) ti = ti + 1;
    
    // parameters
    a = theta[1];
    c = theta[2];
    b = theta[3];
    d = theta[4];
    //a = theta[ti*4 + 1];
    //c = theta[ti*4 + 2];
    //b = theta[ti*4 + 3];
    //d = theta[ti*4 + 4];
    
    // differences
    dS_dt = -a / N * I * S;
    dE_dt = a / N * S * I - c * E;
    dI_dt = c * E - b * I - d * I;
    dR_dt = b * I;
    dD_dt = d * I;
    return {dS_dt, dE_dt, dI_dt, dD_dt, dR_dt};
  }
  
}

data {
  // sizes
  int<lower=1> DAYS; // data size
  int<lower=1> POP; // population
  real<lower=1> TS[DAYS]; // time axis
  
  // parameters
  real<lower=0> prior_a[2]; //[DAYS,2];
  real<lower=0> prior_c[2]; //[DAYS,4];
  real<lower=0> prior_b[2]; //[DAYS,4];
  real<lower=0> prior_d[2]; //[DAYS,4];
  real<lower=0> prior_test[DAYS,2];
  
  // measurements
  real<lower=0> tests[DAYS]; // tests
  real<lower=0,upper=1> confirmed[DAYS]; // positive test ratio
  real<lower=0,upper=1> recovered[DAYS]; // recovered
  real<lower=0,upper=1> deaths[DAYS]; // deaths
  
  // initial solution
  real<lower=0> init_solution[5];
}

transformed data {
  // recovered+deaths
  //real<lower=0> deaths_recovered[DAYS];
  
  // integration data
  real x_r[0]; int x_i[1] = { POP };
  
  // test ratio
  //real<lower=0,upper=1> test_ratio[DAYS];
  //real<lower=0,upper=1> confirmed_mean;
  // posterior test
  //real<lower=0> post_test_alpha;
  //real<lower=0> post_test_beta;
  //real<lower=0> post_test[DAYS,2];
  
  // recovered+deaths
  //for(i in 1:DAYS)
  //  deaths_recovered[i] = deaths[i] + recovered[i];
  // test ratio
  //for(i in 1:DAYS) {
  //  post_test[i,1] = prior_test[1] + confirmed[i] * tests[i];
  //  post_test[i,2] = prior_test[2] + tests[i] - confirmed[i] * tests[i];
    //test_ratio[i] = confirmed[i] / tests[i];
  //}
  //confirmed_mean = sum(test_ratio) / DAYS;
  // posterior test
  //post_test = {
  //  prior_test[1] + confirmed_mean
  //}
  //post_test_alpha = prior_test[1] + confirmed_mean;
  //post_test_beta = prior_test[2] + 1 - confirmed_mean;
  //post_test = {
  //  post_test_alpha / (post_test_alpha + post_test_beta),
  //  post_test_alpha * post_test_beta / pow(post_test_alpha + post_test_beta,2) /
  //    (post_test_alpha + post_test_beta + 1)
  //};
  
}

parameters {
  // latent prior
  real<lower=0,upper=1> a_sir; //[DAYS];
  real<lower=0,upper=1> c_sir; //[DAYS];
  real<lower=0,upper=1> b_sir; //[DAYS];
  real<lower=0,upper=1> d_sir; //[DAYS];
  
  // latent states
  
}

transformed parameters {
  // latent
  real y[DAYS, 5];
  real<lower=0> theta[4];
  //real<lower=0> theta[4*DAYS];
  //for(t in 1:DAYS) {
  //  theta[4*(t-1) + 1] = a_sir[t];
  //  theta[4*(t-1) + 2] = c_sir[t];
  //  theta[4*(t-1) + 3] = b_sir[t];
  //  theta[4*(t-1) + 4] = d_sir[t];
  //}
  theta[1] = a_sir;
  theta[2] = c_sir;
  theta[3] = b_sir;
  theta[4] = d_sir;

  // integration step
  y = integrate_ode_rk45(sir, init_solution, 0, TS, theta, x_r, x_i); //, 1e-2, 1e-2, 100000);
}

model {
  //print("log density before =", target());
  // epidemic
  for(t in 1:DAYS) {
    // priors
    //a_sir[t] ~ beta(prior_a[t,1], prior_a[t,2]);
    //c_sir[t] ~ beta(prior_c[t,1], prior_c[t,2]);
    //b_sir[t] ~ beta(prior_b[t,1], prior_b[t,2]);
    //d_sir[t] ~ beta(prior_d[t,1], prior_d[t,2]);
    a_sir ~ weibull(prior_a[1], prior_a[2]);
    c_sir ~ beta(prior_c[1], prior_c[2]);
    b_sir ~ beta(prior_b[1], prior_b[2]);
    d_sir ~ beta(prior_d[1], prior_d[2]);
    
    // testing
    //print("y[",t,",3] = ", y[t,3]);
    confirmed[t] ~ beta(prior_test[t,1] + tests[t] * y[t,3],
                        prior_test[t,2] + tests[t] * (1 - y[t,3]));
    //confirmed[t] ~ normal(y[t,3]*post_test[1], sqrt(fabs(y[t,3])*post_test[2]) ) T[0,];
    
    // recovered, deaths
    //deaths_recovered[t] ~ normal(y[t,4] * post_test[1], sqrt(y[t,4] * post_test[2]) ) T[0,];
    //deaths[t] ~ normal(y[t,5] * post_test[1], sqrt(y[t,5] * post_test[2]) ) T[0,]; 
    //print("y[",t,",4] = ", y[t,4]);
    deaths[t] ~ normal(y[t,4],1) T[0,];
    recovered[t] ~ normal(y[t,5],1) T[0,];
    //print("y[",t,",5] = ", y[t,5]);
    //recovered[t] ~ normal(fabs(y[t,5])*post_test[1], sqrt(fabs(y[t,5])*post_test[2]) ) T[0,];
  }
  //print("log density after =", target());
}

generated quantities {
  real R0 = a_sir / c_sir;
  real recovery_time = 1 / c_sir;
  //real R0[DAYS];
  //real recovery_time[DAYS];
  //for(t in 1:DAYS) {
  //  R0[t] = a_sir[t] / c_sir[t];
  //  recovery_time[t] = 1 / c_sir[t]; // / POP;
  //}
  //real pred_cases[n_days];
  //pred_cases = neg_binomial_2_rng(col(to_matrix(y), 2) + 1e-5, phi);
}


