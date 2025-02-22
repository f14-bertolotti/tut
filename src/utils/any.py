import click

class Any(click.ParamType):
    def __init__(self, *types):
        self.types = types

    def convert(self, value, param, ctx):
        try: return eval(value)
        except NameError:
            return str(value)
        except Exception: 
            self.fail("Didn't match any of the accepted types.")
