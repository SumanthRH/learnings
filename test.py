"""
Playing around with calculations in the Github stats widget
"""
import numpy as np

# Define the provided exponential cdf function
def exponential_cdf(x):
    return 1 - 2 ** -x

# Define the provided log normal cdf function
def log_normal_cdf(x):
    return x / (1 + x)

# The calculateRank function as provided by the user
def calculate_rank(commits, prs, issues, reviews, stars, followers):
    # Median values and weights for each metric
    COMMITS_MEDIAN, COMMITS_WEIGHT = 250, 2
    PRS_MEDIAN, PRS_WEIGHT = 50, 3
    ISSUES_MEDIAN, ISSUES_WEIGHT = 25, 1
    REVIEWS_MEDIAN, REVIEWS_WEIGHT = 2, 1
    STARS_MEDIAN, STARS_WEIGHT = 50, 4
    FOLLOWERS_MEDIAN, FOLLOWERS_WEIGHT = 10, 1

    # Total weight
    TOTAL_WEIGHT = (COMMITS_WEIGHT + PRS_WEIGHT + ISSUES_WEIGHT +
                    REVIEWS_WEIGHT + STARS_WEIGHT + FOLLOWERS_WEIGHT)

    # Thresholds and levels
    THRESHOLDS = [1, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
    LEVELS = ["S", "A+", "A", "A-", "B+", "B", "B-", "C+", "C"]

    # Calculate the rank
    rank = 1 - (
        COMMITS_WEIGHT * exponential_cdf(commits / COMMITS_MEDIAN) +
        PRS_WEIGHT * exponential_cdf(prs / PRS_MEDIAN) +
        ISSUES_WEIGHT * exponential_cdf(issues / ISSUES_MEDIAN) +
        REVIEWS_WEIGHT * exponential_cdf(reviews / REVIEWS_MEDIAN) +
        STARS_WEIGHT * log_normal_cdf(stars / STARS_MEDIAN) +
        FOLLOWERS_WEIGHT * log_normal_cdf(followers / FOLLOWERS_MEDIAN)
    )/ TOTAL_WEIGHT
    bools = [rank*100 <= thresh for thresh in THRESHOLDS]
    level = LEVELS[bools.index(True)]
    print("rank: ", level)

calculate_rank(commits=260, prs=43, issues=6, followers=43, stars=196, reviews=12)