
functions {

  real[] sir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    // states
    real S = y[1];
    real E = y[2];
    real I = y[3];
    real N = x_i[1];
    
    // parameters
    real a = theta[1];
    real c = theta[2];
    //real b = theta[3];
    //real d = theta[4];
    
    // differences
    real dS_dt = -a * N * I * S;
    real dE_dt = a * N * S * I - c * E;
    real dI_dt = c * E; // - b * I;
    //real dR_dt = b * I;
    //real dD_dt = d * I;
    return {dS_dt, dE_dt, dI_dt}; //, dR_dt};//, dD_dt};
  }
  
}

data {
  // sizes
  int<lower=1> DAYS; // data size
  int<lower=1> POP; // population
  real<lower=1> TS[DAYS]; // time axis
  
  // parameters
  real<lower=0> prior_a[2];
  real prior_c[4];
  //real prior_b[4];
  //real<lower=0> prior_d[2];
  real<lower=0> prior_test[2];
  
  // measurements
  real<lower=0,upper=1> tests[DAYS]; // tests
  real<lower=0,upper=1> confirmed[DAYS]; // positive tests
  //real<lower=0,upper=1> recovered[DAYS]; // recovered
  //real<lower=0,upper=1> deaths[DAYS]; // deaths
  
  // initial solution
  real<lower=0> init_solution[3];
}

transformed data {
  // recovered+deaths
  //real<lower=0> deaths_recovered[DAYS];
  
  // integration data
  real x_r[0]; int x_i[1] = { POP };
  
  // test ratio
  real<lower=0,upper=1> test_ratio[DAYS];
  real<lower=0,upper=1> confirmed_mean;
  // posterior test
  real<lower=0> post_test_alpha;
  real<lower=0> post_test_beta;
  real post_test[2];
  
  // recovered+deaths
  //for(i in 1:DAYS)
  //  deaths_recovered[i] = deaths[i] + recovered[i];
  // test ratio
  for(i in 1:DAYS)
    test_ratio[i] = confirmed[i] / tests[i];
  confirmed_mean = sum(test_ratio) / DAYS;
  // posterior test
  post_test_alpha = prior_test[1] + POP*confirmed_mean;
  post_test_beta = prior_test[2] + POP - POP*confirmed_mean;
  post_test = {
    post_test_alpha / (post_test_alpha + post_test_beta),
    post_test_alpha * post_test_beta / pow(post_test_alpha + post_test_beta,2) /
      (post_test_alpha + post_test_beta + 1)
  };
}

parameters {
  // latent prior
  real<lower=0> a_sir;
  real<lower=0> c_sir_unscaled;
  //real<lower=0> b_sir_unscaled;
  //real<lower=0> d_sir;
  
  // test prior
  real<lower=0> p_test;
}

transformed parameters {
  // transformed
  real<lower=0> c_sir = (c_sir_unscaled - prior_c[3]) / prior_c[4];
  //real<lower=0> b_sir = (b_sir_unscaled - prior_b[3]) / prior_b[4];
  
  // latent
  real y[DAYS, 3];
  real theta[2];
  theta[1] = a_sir;
  theta[2] = c_sir;
  //theta[3] = b_sir;
  //theta[4] = d_sir;

  // integration step
  y = integrate_ode_rk45(sir, init_solution, 0, TS, theta, x_r, x_i); //, 1e-2, 1e-2, 100000);
}

model {
  // epidemic
  for(t in 1:DAYS) {
    // priors
    a_sir ~ beta(prior_a[1], prior_a[2]);
    //a_sir ~ uniform(prior_a[1], prior_a[2]);
    c_sir_unscaled ~ beta(prior_c[1], prior_c[2]);
    //b_sir_unscaled ~ beta(prior_b[1], prior_b[2]);
    //d_sir ~ uniform(prior_d[1], prior_d[2]);
    
    // testing
    confirmed[t] ~ normal(y[t,3] * post_test[1], sqrt(y[t,3] * post_test[2]) ) T[0,];
    
    // recovered, deaths
    //deaths_recovered[t] ~ normal(y[t,4] * post_test[1], sqrt(y[t,4] * post_test[2]) ) T[0,];
    //deaths[t] ~ normal(y[t,5] * post_test[1], sqrt(y[t,5] * post_test[2]) ) T[0,]; 
    //deaths[t] ~ normal(y[t,4],1) T[0,];
    //recovered[t] ~ normal(y[t,5],1) T[0,];
  }
}

generated quantities {
  real R0 = a_sir / c_sir;
  real recovery_time = 1 / c_sir / POP;
  //real pred_cases[n_days];
  //pred_cases = neg_binomial_2_rng(col(to_matrix(y), 2) + 1e-5, phi);
}


