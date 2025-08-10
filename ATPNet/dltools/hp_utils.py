import inspect
from argparse import ArgumentParser, Namespace
from typing import Dict, Union, List, Tuple, Any


class PropertiesTool:
    def __init__(self, *args, **kwargs):
        super().__init__()

        frame = inspect.currentframe().f_back
        init_args = self._get_init_args(frame)
        self.hparams = init_args

    @staticmethod
    def _get_init_args(frame) -> dict:
        _, _, _, local_vars = inspect.getargvalues(frame)
        if '__class__' not in local_vars:
            return {}
        cls = local_vars['__class__']
        init_parameters = inspect.signature(cls.__init__).parameters
        local_args = {k: local_vars[k] for k in init_parameters.keys() if k != 'self'}
        return local_args

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, filter_args: List[str] = []) -> ArgumentParser:
        module_name = cls.__class__.__name__

        parser = ArgumentParser(parents=[parent_parser], add_help=False, )
        valid_kwargs = inspect.signature(cls.__init__).parameters

        filter_args.append('self')
        for k in valid_kwargs:
            if k in filter_args:
                continue

            v = valid_kwargs.get(k).default
            if v is inspect.Parameter.empty:
                parser.add_argument(f'--{k}', help=f'the parameter of {module_name}_{k}')
            else:
                v_type = type(v)
                parser.add_argument(f'--{k}', default=v, type=v_type,
                                    help=f'the parameter of {module_name}_{k}')

        return parser

    @classmethod
    def from_argparse_args(cls, args: Namespace, **kwargs):
        params = vars(args)

        # we only want to pass in valid Trainer args, the rest may be user specific
        valid_kwargs = inspect.signature(cls.__init__).parameters
        cls_kwargs = dict((k, params[k]) for k in valid_kwargs if k in params)
        cls_kwargs.update(**kwargs)

        return cls(**cls_kwargs)


def PropertiesToolDecorator(cls):
    cls_init = cls.__init__

    def init(self, *args, **kwargs):
        # print('Create class \'{}\' by PropertiesTool'.format(cls.__name__))

        hparams = {}
        valid_kwargs = dict(inspect.signature(cls_init).parameters)
        valid_kwargs.pop('self')
        for i, k in enumerate(valid_kwargs):
            if i < len(args):
                v = args[i]
            elif k in kwargs.keys():
                v = kwargs.get(k)
            else:
                v = valid_kwargs.get(k).default

            if v is inspect.Parameter.empty:
                raise TypeError('__init__() got multiple values for argument \'{}\''.format(k))

            hparams.update({k: v})

        self.hparams = hparams
        cls_init(self, *args, **kwargs)

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, filter_args: List[str] = []) -> ArgumentParser:
        module_name = cls.__class__.__name__

        parser = ArgumentParser(parents=[parent_parser], add_help=False, )
        valid_kwargs = inspect.signature(cls_init).parameters

        filter_args.append('self')
        for k in valid_kwargs:
            if k in filter_args:
                continue

            v = valid_kwargs.get(k).default
            if v is inspect.Parameter.empty:
                parser.add_argument(f'--{k}', help=f'the parameter of {module_name}_{k}')
            else:
                v_type = type(v)
                parser.add_argument(f'--{k}', default=v, type=v_type,
                                    help=f'the parameter of {module_name}_{k}')

        return parser

    @classmethod
    def from_argparse_args(cls, args: Namespace, **kwargs):
        params = vars(args)

        # we only want to pass in valid Trainer args, the rest may be user specific
        valid_kwargs = inspect.signature(cls_init).parameters
        cls_kwargs = dict((k, params[k]) for k in valid_kwargs if k in params)
        cls_kwargs.update(**kwargs)

        return cls(**cls_kwargs)

    cls.__init__ = init
    cls.add_argparse_args = add_argparse_args
    cls.from_argparse_args = from_argparse_args

    return cls


class __PropertiesToolDecorator_bk:
    def __init__(self, cls):
        self._cls = cls
        self.__name__ = self._cls.__name__

    def __call__(self, *args, **kwargs):
        print('Create class \'{}\' by PropertiesTool'.format(self._cls.__name__))

        hparams = {}
        valid_kwargs = dict(inspect.signature(self._cls.__init__).parameters)
        valid_kwargs.pop('self')
        for i, k in enumerate(valid_kwargs):
            if i < len(args):
                v = args[i]
            elif k in kwargs.keys():
                v = kwargs.get(k)
            else:
                v = valid_kwargs.get(k).default

            if v is inspect.Parameter.empty:
                raise TypeError('__init__() got multiple values for argument \'{}\''.format(k))

            hparams.update({k: v})

        obj = self._cls(*args, **kwargs)
        obj.hparams = hparams
        return obj

    def add_argparse_args(self, parent_parser: ArgumentParser, filter_args: List[str] = []) -> ArgumentParser:
        module_name = self._cls.__class__.__name__

        parser = ArgumentParser(parents=[parent_parser], add_help=False, )
        valid_kwargs = inspect.signature(self._cls.__init__).parameters

        filter_args.append('self')
        for k in valid_kwargs:
            if k in filter_args:
                continue

            v = valid_kwargs.get(k).default
            if v is inspect.Parameter.empty:
                parser.add_argument(f'--{k}', help=f'the parameter of {module_name}_{k}')
            else:
                v_type = type(v)
                parser.add_argument(f'--{k}', default=v, type=v_type,
                                    help=f'the parameter of {module_name}_{k}')

        return parser

    def from_argparse_args(self, args: Namespace, **kwargs):
        params = vars(args)

        # we only want to pass in valid Trainer args, the rest may be user specific
        valid_kwargs = inspect.signature(self._cls.__init__).parameters
        cls_kwargs = dict((k, params[k]) for k in valid_kwargs if k in params)
        cls_kwargs.update(**kwargs)

        return self.__call__(**cls_kwargs)
