import argparse
import logging
import os
import sys
import traceback
import yaml
from easyDataverse import  Dataset

__author__ = "NilsWildt"
__copyright__ = "NilsWildt"
__license__ = "MIT"
__version__ = "0.1.0"

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stdout,
)
_logger = logging.getLogger(__name__)


def download_data(datadir, DOI, KEY=None, URL="https://darus.uni-stuttgart.de",):
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    try:
        _logger.debug("Start downloading. This may take a while.")
        # Dataset.from_dataverse_doi(doi=self.DOI, filedir=str(self.datadir), dataverse_url=self.URL)
        dataset = Dataset.from_dataverse_doi(
                        filedir = datadir,
                        doi = DOI,
                        api_token = KEY,
                        dataverse_url = URL,
                        )
        print(dataset)
        
    except Exception:
        _logger.info(traceback.format_exc())
    

def parse_args(args):
    """Parse command line parameters

    Args:
    args (List[str]): command line parameters as list of strings
    (for example  ``["--help"]``).

    Returns:
    :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Geostatistical Inversion Benchmarking")

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="./configs/config.yaml",
        help=
        f"Choose a configfile. The folders will be setup and used as specified in the Config file. Default is in './configs/config.yaml'"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Geostatistical Inversion Benchmarking {ver}".format(
            ver=__version__),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
    loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel,
                        stream=sys.stdout,
                        format=logformat,
                        datefmt="%Y-%m-%d %H:%M:%S")

def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)
    if args.config:
        # _logger.info(f"The used config is {args.config}.")
        # benchmark = geostatbench.BayesInvBench(args.config)
        # _logger.info(f"Chose scenario {benchmark.scenario_name}")
        # benchmark.download_data_in_background()
        # _logger.info("Finished data download.")

        _logger.info(f"The used config is {args.config}.")
        with open(args.config, 'r') as config_file:
            config = yaml.safe_load(config_file)
        
        data_path_r = config.get('datapath') # relative path
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), data_path_r)

        download_data(datadir=data_path,
                        DOI=config.get('dataset_doi'),
                        KEY=config.get('api_key'),
                        URL="https://darus.uni-stuttgart.de",
                        )
        
        _logger.info("Finished data download.")
        
    else: 
        _logger.warning("No config file was given. Please use -c to specify a config file.")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])

if __name__ == "__main__":
    _logger.info("Start in main")
    run()