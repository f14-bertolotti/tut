import utils
import click

class Any(click.ParamType):
    def __init__(self, *types):
        self.types = types

    def convert(self, value, param, ctx):
        for type in self.types:
            if type == bool: type = utils.str2bool
            try: return type(value)
            except ValueError: continue
        self.fail("Didn't match any of the accepted types.")
