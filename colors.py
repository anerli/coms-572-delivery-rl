# https://www.lihaoyi.com/post/BuildyourownCommandLinewithANSIescapecodes.html
class Colors:
    # Blue-ish BG, black FG
    PLAYER = '\u001b[31m'#'\u001b[48;2;40;180;255;38;2;0;0;0m'
    # Yellowish BG, black FG
    PICKUP = '\u001b[33m'#'\u001b[48;2;200;200;0;38;2;0;0;0m'
    # Dark green BG, black FG
    DROPOFF = '\u001b[32m'#'\u001b[48;2;0;200;0;38;2;0;0;0m'

    RESET = '\u001b[0m'