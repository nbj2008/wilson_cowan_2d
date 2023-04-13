from itertools import cycle
from IPython.display import clear_output

_dots = ["⢀⠀", "⡀⠀", "⠄⠀", "⢂⠀", "⡂⠀", "⠅⠀", "⢃⠀", "⡃⠀", "⠍⠀", "⢋⠀", "⡋⠀",
         "⠍⠁", "⢋⠁", "⡋⠁", "⠍⠉", "⠋⠉", "⠋⠉", "⠉⠙", "⠉⠙", "⠉⠩", "⠈⢙", "⠈⡙",
         "⢈⠩", "⡀⢙", "⠄⡙", "⢂⠩", "⡂⢘", "⠅⡘", "⢃⠨", "⡃⢐", "⠍⡐", "⢋⠠", "⡋⢀",
         "⠍⡁", "⢋⠁", "⡋⠁", "⠍⠉", "⠋⠉", "⠋⠉", "⠉⠙", "⠉⠙", "⠉⠩", "⠈⢙", "⠈⡙",
         "⠈⠩", "⠀⢙", "⠀⡙", "⠀⠩", "⠀⢘", "⠀⡘", "⠀⠨", "⠀⢐",  "⠀⡐", "⠀⠠", "⠀⢀",
         "⠀⡀"]


def make_counter(end, rate=20, wait=True):
    e = end
    counter = 0
    dot_cycle = cycle(_dots)

    def count_time(t):
        nonlocal counter
        counter += 1
        if counter % rate == 0:
            clear_output(wait=wait)
            print(next(dot_cycle), f"Time Step: {int(t):0>4}/{int(e):0>4}",
                  f"Percent Done: {t/e:>.1%}")

    return count_time
