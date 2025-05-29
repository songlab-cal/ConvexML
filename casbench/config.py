from inspect import signature
from typing import Any, Callable, Dict, List, Tuple, Union

# A config is just the class name (or function name) followed by the list of
# arguments.
Config = Tuple[str, List[Tuple[str, Any]]]


def create_config_from_dict(config_dict: Dict) -> Config:
    if sorted(list(config_dict.keys())) != ["args", "identifier"]:
        raise ValueError(f'config_dict should have keys ["args", "identifier"].\nconfig_dict={config_dict}')
    identifier = config_dict["identifier"]
    args_dict = config_dict["args"]
    config = (identifier, sorted(args_dict.items()))
    return config


def sanity_check_config(config: Config):
    identifier, args = config
    for i in range(len(args) - 1):
        if args[i][0] >= args[i + 1][0]:
            raise ValueError(
                "Arguments of Config should be sorted in increasing alphabetic "
                f"order. Found '{args[i][0]}' before '{args[i + 1][0]}'. "
                f"Config: {config}"
            )


def smart_call(func: Callable, config_dicts: Dict[str, Union[Dict, Config]]):
    """
    Given a cached function (which takes in Config arguments), call it with
    a simpler API: Just Dict in place of Configs. The Dicts will get
    automatically converted to Configs. Moreover, excess Configs that are
    not part of the function signature are dropped.

    One can also pass in Configs directly, in which case they are simply
    subsetted.
    """
    sig = signature(func)
    configs = {
        arg_name: (
            create_config_from_dict(config_dict)
            if type(config_dict) is dict
            else config_dict
        )
        for (arg_name, config_dict) in config_dicts.items()
        if arg_name in sig.parameters
    }
    return func(**configs)
