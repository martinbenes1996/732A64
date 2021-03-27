
functions {

  real[] sir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {
    // states
    real S = y[1];
    real E = y[2];
    real I = y[3];
    real POP = x_i[1];
    int WINDOW = x_i[2];
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
    int ti = 1;
    while(ti < floor(t)) ti = ti + 1;
    ti = ti / WINDOW + 1;
    
    // parameters
    a = theta[(ti-1)*4 + 1];
    c = theta[(ti-1)*4 + 2];
    b = theta[(ti-1)*4 + 3];
    d = theta[(ti-1)*4 + 4];
    
    // differences
    dS_dt = -a / POP * S * I;
    dE_dt = a / POP * S * I - c * E;
    dI_dt = c * E - b * I - d * I;
    dR_dt = b * I;
    dD_dt = d * I;
    return {dS_dt, dE_dt, dI_dt, dR_dt, dD_dt};
  }
}

data {
  // sizes
  int<lower=1> DAYS; // data size
  int<lower=1> WINDOW; // day window
  int<lower=1> POP; // population
  real<lower=0,upper=1> INDIV; // ratio of single person
  real<lower=1> TS[DAYS]; // time axis
  
  // parameters
  real<lower=0> prior_a[2];
  real<lower=0> prior_c[2];
  real<lower=0> prior_b[2];
  real<lower=0> prior_d[2];
  real<lower=0> prior_test[2];
  real<lower=0> prior_test_rec[2];
  real<lower=0> prior_deaths[2];
  
  // measurements
  real<lower=0> tests[DAYS]; // tests
  real<lower=0,upper=1> confirmed[DAYS]; // positive test ratio
  real<lower=0> recovered[DAYS]; // recovered
  real<lower=0> deaths[DAYS]; // deaths
  
  // initial solution
  real<lower=0> init_solution[5];
}

transformed data {
  // day window
  int DAYS_W = DAYS / WINDOW + 1;
  
  // integration data
  real x_r[0]; int x_i[2] = { POP, WINDOW }; //, DAYS };
}

parameters {
  // latent prior
  real<lower=0,upper=1> a_sir[DAYS_W];
  real<lower=0,upper=1> c_sir[DAYS_W];
  real<lower=0,upper=1> b_sir[DAYS_W];
  real<lower=0,upper=1> d_sir[DAYS_W];
  
  // latent states
}

transformed parameters {
  // latent
  real y_[DAYS, 5];
  real<lower=0> y[DAYS, 5];
  
  //real<lower=0> theta[4];
  //theta[1] = a_sir;
  //theta[2] = c_sir;
  //theta[3] = b_sir;
  //theta[4] = d_sir;
  
  real<lower=0,upper=1> theta[4*DAYS_W];
  //print("Total: ", 4*DAYS_W);
  for(t in 1:DAYS_W) {
    theta[4*(t-1) + 1] = a_sir[t];
    theta[4*(t-1) + 2] = c_sir[t];
    theta[4*(t-1) + 3] = b_sir[t];
    theta[4*(t-1) + 4] = d_sir[t];
  }
  
  //print(4*DAYS_W);

  // integration step
  y_ = integrate_ode_rk45(sir, init_solution, 0, TS, theta, x_r, x_i);
  y = fabs(y_);
  //for(d in 1:DAYS)
  //  for(i in 1:5)
  //    y = fabs(y);
  // saturation arithmetics
  //for(d in 1:DAYS)
  //  for(i in 1:5)
  //    if(y[d,i] <= 0.)
  //      y[d,i] = INDIV;
}

model {
  for(t in 1:DAYS_W) {
    // priors
    a_sir[t] ~ weibull(prior_a[1], prior_a[2]);
    #a_sir[t] ~ beta(prior_a[1], prior_a[2]);
    c_sir[t] ~ beta(prior_c[1], prior_c[2]);
    b_sir[t] ~ beta(prior_b[1], prior_b[2]);
    d_sir[t] ~ beta(prior_d[1], prior_d[2]);
  }
  //a_sir ~ weibull(prior_a[1], prior_a[2]);
  //c_sir ~ beta(prior_c[1], prior_c[2]);
  //b_sir ~ beta(prior_b[1], prior_b[2]);
  //d_sir ~ beta(prior_d[1], prior_d[2]);
  
  // epidemic
  for(t in 1:DAYS) {//
    real densBefore = target();
    //print("log density: ", target());
    
    // testing
    //print(t, "/", DAYS, ") ", y[t,2:3], " - ", tests[t]);
    confirmed[t] ~ beta(prior_test[1] + tests[t] * y[t,3],
                        prior_test[2] + tests[t] * (1 - y[t,3]));
    //confirmed[t] ~ normal(y[t,3]*post_test[1], sqrt(fabs(y[t,3])*post_test[2]) ) T[0,];
    //print("after confirmed: ", target());
    
    print("x = ", confirmed[t], "; ",//"prior_test = ", prior_test[t,1:2], "; ",
          "post_test = [", prior_test[1] + tests[t] * y[t,3],",",
                           prior_test[2] + tests[t] * (1 - y[t,3]),"]; ",
          "tests = ", tests[t], "; ",
          "y = ", y[t,3], "; ",
          "dens = ", densBefore, "->", target());
   
    // recovered
    //recovered[t] ~ normal(y[t,4] * POP, 1) T[0,];
    recovered[t] ~ beta(prior_test_rec[1] + tests[t] * y[t,4],
                        prior_test_rec[2] + tests[t] * (1 - y[t,4]));
    print("recovered: ", recovered[t], "; prior beta(", prior_test_rec, ");",
          "post beta(", prior_test_rec[1] + tests[t] * fabs(y[t,4]), ",",
                        prior_test_rec[2] + tests[t] * (1 - fabs(y[t,4])),")")
    //print("after recovered: ", target());
    
    //recovered[t] ~ normal(y[t,5],1/POP) T[0,];
    // deaths
    deaths[t] ~ beta(prior_deaths[1] + tests[t] * y[t,5],
                     prior_deaths[2] + tests[t] * (1 - y[t,5]));
    //deaths[t] ~ normal(y[t,5] * POP, 1) T[0,];
    //print("after deaths: ", target());
    
    //print("confirmed ~ beta(",prior_test[1] + tests[t] * y[t,3], ",",
    //                          prior_test[2] + tests[t] * (1 - y[t,3]),");",
    //      "dens = ",densBefore, " -> ", target());

  }
}

generated quantities {
  //real R0 = a_sir / c_sir;
  //real recovery_time = 1 / c_sir;
  real R0[DAYS_W];
  real recovery_time[DAYS_W];
  //int ti;
  for(ti in 1:DAYS_W) {
    //ti = t / WINDOW + 1;
    R0[ti] = a_sir[ti] / c_sir[ti];
    recovery_time[ti] = 1 / c_sir[ti]; // / POP;
  }
  //real pred_cases[n_days];
  //pred_cases = neg_binomial_2_rng(col(to_matrix(y), 2) + 1e-5, phi);
}


