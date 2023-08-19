from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        # Add Style Transfer options
        parser.add_argument('--results_dir', type=str,
                            default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str,
                            default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true',
                            help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=100,
                            help='how many test images to run')
        parser.add_argument('--original_domain', type=str, default='A',
                            help='original domain (can be either A or B)')

        # Add YeaZ options
        parser.add_argument('--path_to_yeaz_weights', default=None,
                            type=str, help="Specify YeaZ weights path.")
        parser.add_argument('--threshold', default=0.5,
                            type=float, help="Specify threshold value.")
        parser.add_argument('--min_seed_dist', default=5, type=int,
                            help="Specify minimum distance between seeds.")
        parser.add_argument('--min_epoch', default=1,
                            type=int, help="Specify min epoch.")
        parser.add_argument('--max_epoch', default=200,
                            type=int, help="Specify max epoch.")
        parser.add_argument('--epoch_step', default=1,
                            type=int, help="Specify epoch step.")
        parser.add_argument('--metrics_path', default=None,
                            type=str, help="Specify where to save metrics path.")

        # add metrics options metrics_patch_borders that accepts a tuple
        # you can specify a tuple in the command line like this:
        # --metrics_patch_borders 0 0 0 0
        parser.add_argument('--metrics_patch_borders', nargs='+', type=int,
                            default=None, help="Specify patch borders.")

        

        # Add general options
        parser.add_argument('--skip_style_transfer',
                            action='store_true', help='Skip style transfer.')
        parser.add_argument('--skip_segmentation',
                            action='store_true', help='Skip segmentation.')
        parser.add_argument(
            '--skip_metrics', action='store_true', help='Skip metrics.')
        parser.add_argument(
            '--plot_metrics', action='store_true', help='Plot metrics.')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
