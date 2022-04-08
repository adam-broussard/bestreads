from matplotlib import pyplot as plt


def testplot():
    """
    This is a test plot to make sure that linting, pushing, and branch control
    are all working
    """

    # Adding a commit to see if things run correctly...

    x = [1, 2, 3]
    y = x

    fig = plt.figure(figsize=(8, 8))
    sp = fig.add_subplot(111)

    sp.plot(x, y)
    sp.set_xlabel('X')
    sp.set_ylabel('Y')
