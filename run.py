import matplotlib.pyplot as plt
import sys
sys.path.append('src')

# === prior ===
import prior
# prior c (EI)
prior.plot_EI(save = True)
plt.show()
# prior b (IR)
prior.plot_IR(save = True)
plt.show()
# parameters for the prior distributions
prior_parameters = prior.priors(save = True)
print(prior_parameters)

# === lethality ===
import lethality
# age-gender distribution
lethality.plot_violin(save = True)
plt.show()
# test that deaths are > 60 years
over60 = lethality.test_over60()
print(over60)