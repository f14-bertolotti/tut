import click

class Any(click.ParamType):
    def __init__(self, *types):
        self.types = types

    def convert(self, value, param, ctx):
        for type in self.types:
            try: return type(value)
            except ValueError: continue
        self.fail("Didn't match any of the accepted types.")
