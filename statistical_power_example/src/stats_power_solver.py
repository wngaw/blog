import sys
import argparse
import math
from statsmodels.stats.power import TTestIndPower, NormalIndPower


def solve_statistical_power_z_test(mean_exp=50, mean_ctrl=40, sd_exp=4, sd_ctrl=3, num_exp=200, num_ctrl=40, alpha=0.05):
    """Solve for statistical_power for z-test, given effect_size and sample_size. Assumes n is large (>= 30).
    When to use: Typically used post-experiment to estimate significance of experiment
    Params:
        mean_exp: metric mean for the exp group. E.g. mean of sales_revenue of the exp group.
        mean_ctrl: metric mean of the ctrl group. E.g. mean of sales_revenue of the ctrl group.
        sd_exp: metric standard deviation of the exp group. E.g. sd of sales_revenue of exp group.
        sd_ctrl: metric standard deviation of the exp group. E.g. sd of sales_revenue of ctrl group.
        num_exp: number of observation in exp group.
        num_ctrl: number of observation in ctrl group.
        alpha: probability of a type I error; wrong rejections if the H0 is true.
    Returns:
        statistical_power: the probability that the test correctly rejects H0
    """

    # Calculate effect size
    pooled_sd = math.sqrt(((num_exp - 1) * (sd_exp**2) +
                           (num_ctrl - 1) * (sd_ctrl**2)) /
                          (num_exp + num_ctrl - 2))
    abs_effect_size = abs(mean_exp - mean_ctrl) / pooled_sd  # absolute to account for both positive and negative result.

    # Parameters for solver
    effect_size = abs_effect_size  # cohen's d": standardized effect size, difference between the two means divided by sd.
    alpha = alpha
    nobs1 = num_exp  # number of observations of sample 1. nobs_2 = ratio x nobs_1
    ratio = num_ctrl/num_exp  # ratio of nobs in sample 2 relative to sample 1
    alternative = 'two-sided'  # power is calculated for a two-sided test.

    # Power solver
    z_test_power_solver = NormalIndPower()
    statistical_power = z_test_power_solver.solve_power(effect_size=effect_size, nobs1=nobs1, alpha=alpha,
                                                        ratio=ratio, alternative=alternative, power=None)
    return statistical_power


def solve_sample_size_z_test(abs_effect_size=0.1, ctrl_exp_ratio=0.2, statistical_power=0.8, alpha=0.05):
    '''Solve for sample_size for z-test, given effect_size and statistical_power. Assumes n is large (>= 30).
    When to use: Typically used pre-experiment to estimate min sample size
    Params:
        abs_effect_size: range from 0 to 3. Default value of 0.1 is a very conservative test performance where there is
                         only 53% probability that that person from exp will be higher than person from ctrl.
                         In reality, you will never know this number before the experiment.
                         Hence, a conservative input is most appropriate.
                         For interpretation: https://www.leeds.ac.uk/educol/documents/00002182.htm
        ctrl_exp_ratio: ratio of num_ctrl over num_exp
        statistical_power: the probability that the test correctly rejects H0. Usuualy set to 0.8.
        alpha: probability of a type I error; wrong rejections if the H0 is true.
    Return:
        num_exp: num_exp: number of observation in exp group.
        num_ctrl: number of observation in ctrl group.
    '''

    alpha = alpha
    alternative = 'two-sided'  # power is calculated for a two-sided test.

    # Power solver
    z_test_power_solver = NormalIndPower()
    num_exp = z_test_power_solver.solve_power(effect_size=abs_effect_size, nobs1=None, alpha=alpha,
                                              ratio=ctrl_exp_ratio, alternative=alternative, power=statistical_power)
    num_ctrl = ctrl_exp_ratio * num_exp
    return num_exp, num_ctrl


def solve_abs_effect_size_z_test(num_exp=200, num_ctrl=40, statistical_power=0.8, alpha=0.05):
    '''Solve for effect_size for z-test, given sample_size and statistical_power. Assumes n is large (>= 30).
    When to use: Typically used pre-experiment to estimate effect_size requred to reject H0
    Params:
        num_exp: number of observation in exp group.
        num_ctrl: number of observation in ctrl group.
        statistical_power: the probability that the test correctly rejects H0
        alpha: probability of a type I error; wrong rejections if the H0 is true.
    Return:
        abs_effect_size: range from 0 to 3. For interpretation: https://www.leeds.ac.uk/educol/documents/00002182.htm
    '''

    alpha = alpha  # probability of a type I error; wrong rejections if the H0 is true
    nobs1 = num_exp  # number of observations of sample 1. nobs_2 = ratio x nobs_1
    ratio = num_ctrl/num_exp  # ratio of nobs in sample 2 relative to sample 1
    alternative = 'two-sided'  # power is calculated for a two-sided test.

    # Power solver
    z_test_power_solver = NormalIndPower()
    effect_size = z_test_power_solver.solve_power(effect_size=None, nobs1=nobs1, alpha=alpha, ratio=ratio,
                                                  alternative=alternative, power=statistical_power)
    abs_effect_size = abs(effect_size)
    return abs_effect_size


def main(args):
    if args.parameter == 'stats_power':
        statistical_power = solve_statistical_power_z_test()
        print("statistical power: %0.3f" % (statistical_power))

    elif args.parameter == 'sample_size':
        num_exp, num_ctrl = solve_sample_size_z_test()
        print("minimum sample size for experimental group: %0.f" % (num_exp))
        print("minimum sample size for control group: %0.f" % (num_ctrl))

    elif args.parameter == 'abs_effect_size':
        abs_effect_size = solve_abs_effect_size_z_test()
        print("required abs_effect_size to reject H0: %0.3f" % (abs_effect_size))

    else:
        print('Please input parameter: [statistical_power, sample_size, abs_effect_size].')


def parse_args(args):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # Keyword arguments
    parser.add_argument('-param', '--parameter', type=str, action='store', dest='parameter', default='stats_power',
                        help='statistical_power, sample_size, abs_effect_size')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
