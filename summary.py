from argparse import ArgumentParser
from numpy import mean, std, nanmean
from pickle import load


levels = {
    'Easy': {
        'min_difficulty': 0.0,
        'max_difficulty': 0.35,
    },
    'Med': {
        'min_difficulty': 0.35,
        'max_difficulty': 0.45,
    },
    'Hard': {
        'min_difficulty': 0.45,
        'max_difficulty': 0.5,
    }
}


difficulties = [0., 1e-6, 0.25, 0.5]
speeds = [0.01, 0.05, 0.10, 0.15]
max_ur5s_count = 10
min_ur5s_count = 1

def print_summary_detail(scores):
    success_rates = [score['success'] for score in scores]
    print('| # Arms | ', end='')
    for level in levels.keys():
        print(level + '\t| ', end='')
    print('Avg (Count) \t| Collision Failure Rate|')
    print('-'*73)

    for ur5_count in range(1, 11):
        output = '| {}\t |'.format(ur5_count)
        ur5_count_success_rates = [
            score for score in scores
            if score['task']['ur5_count'] == ur5_count]
        for level in levels.values():
            min_difficulty = level['min_difficulty']
            max_difficulty = level['max_difficulty']
            filtered_scores = [
                score['success'] for score in ur5_count_success_rates
                if score['task']['difficulty'] >= min_difficulty
                and score['task']['difficulty'] <= max_difficulty]
            if len(filtered_scores) == 0:
                output += ' -\t|'
            else:
                output += '{:.03f}\t|'.format(nanmean(filtered_scores))
        tmp = [score['success']
               for score in ur5_count_success_rates]
        output += '{:.03f} ({})\t|'.format(nanmean(tmp), len(tmp))
        collision_failure_rate = [
            1 if score['collision'] > 0 else 0
            for score in scores
            if score['task']['ur5_count'] == ur5_count
            and score['success'] == 0]
        output += f'\t{nanmean(collision_failure_rate):.03f}\t\t|'\
            if len(collision_failure_rate) else '\t-\t\t|'
        print(output)


def print_summary(benchmark_scores):
    for key in benchmark_scores[0].keys():
        if key == 'task' or key == 'debug':
            continue
        scores = [score[key] for score in benchmark_scores]
        print(key)
        print('\tmean:', nanmean(scores))
        print('\tstd:', std(scores))


if __name__ == "__main__":
    parser = ArgumentParser("Benchmark Score Summarizer")
    parser.add_argument("file",
                        type=str,
                        help='path to benchmark score pickle file')
    args = parser.parse_args()
    benchmark_scores = load(open(args.file, 'rb'))
    print_summary(benchmark_scores)
    print_summary_detail(benchmark_scores)
