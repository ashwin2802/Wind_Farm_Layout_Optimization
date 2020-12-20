from simann import Annealer

if __name__ == "__main__":
    iters = int(input())
    ann = Annealer(max_iterations=iters)
    ann.run()