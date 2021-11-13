from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False

        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')
        parser.set_defaults(pool_size=0, gan_mode='lsgan')
        parser.add_argument('--lambda_L1', type=float, default=500, help='weight for L1 loss')

        parser.add_argument('--cfg', type=str, default='config.yaml',
                            help='configuration of tracking')
        parser.add_argument('--dataset', type=str, default='OTB100',
                            help='datasets')
        parser.add_argument('--dataset_dir', type=str,
                            default='/media/wattanapongsu/4T/dataset',
                            help='dataset directory')
        parser.add_argument('--video', default='', type=str,
                            help='eval one special video')
        parser.add_argument('--saved_dir', default='', type=str,
                            help='save images and videos in this directory')
        parser.add_argument('--fabricated_dir', default='', type=str,
                            help='save images and videos in this directory')
        parser.add_argument('--netG_pretrained', default='', type=str,
                            help='netG pretrained ')
        parser.add_argument('--snapshot', default='', type=str,
                            help='snapshot of models to eval')
        parser.add_argument('--attack_search', action='store_true',
                            help='enable attack search')
        parser.add_argument('--model_name', default='', type=str,
                            help='model name ')
        parser.add_argument('--k', default=1, type=float,
                            help='disturbed parameter in template')
        parser.add_argument('--ks', default=1, type=float,
                            help='amplified parameter in searching')
        parser.add_argument('--chk', default=1, type=int,
                            help='checkpoint number')
        parser.add_argument('--export_video', action='store_true',
                            help='export video output')
        parser.add_argument('--vis', dest='vis', action='store_true')
        parser.add_argument('--gpus', default=0, type=int,
                            help='number of gpus')
        parser.add_argument('--search_attack', action='store_true',
                            help='attack search image')
        parser.add_argument('--z_size', type=int, default=128,
                            help='interpolate template size')
        parser.add_argument('--model_search', action='store_true',
                            help='generate noise in search style')

        return parser