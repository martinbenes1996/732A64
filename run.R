
source('src/R0.R')

# plot serial interval distribution
plot_serial_interval(show.hist = T)

# time-dependent R0 estimate from confirmed
plot_R0_series('CZ','2020-03-10')
plot_R0_series('PL','2020-03-10')
plot_R0_series('SE','2020-03-01')
plot_R0_series('IT','2020-03-01')
# time dependent R0 estimate from tests and confirmed
plot_tests_R0_series('CZ','2020-03-10')
plot_tests_R0_series('PL','2020-03-10')
plot_tests_R0_series('SE','2020-03-01')
plot_tests_R0_series('IT','2020-03-01')
# R0 box plots per month
plot_R0_box(c('CZ','IT','PL','SE'), '2020-03-25','2021-02-01')
plot_R0_box(c('CZ','IT','PL','SE'), '2020-03-01','2020-03-31')
# R0 box trace plot
plot_trace_R0_box(c('CZ','IT','PL','SE'), '2020-03-20')
