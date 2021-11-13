from pix2pix.options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')

        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        # parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        parser.set_defaults(norm='batch', netG='unet_128', dataset_mode='aligned')
        parser.set_defaults(pool_size=0, gan_mode='lsgan')
        parser.add_argument('--l1', type=float, default=500, help='weight for L1 loss')
        parser.add_argument('--l2', type=float, default=0.001, help='weight for l2 loss')
        parser.add_argument('--reg', type=float, default=1, help='weight for reg loss')
        parser.add_argument('--cls', type=float, default=0.1, help='weight for cls loss')
        parser.add_argument('--droprate', default='10', type=float, help='droprate')

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
        parser.add_argument('--epsilon', default='0.1', type=float,
                            help='noise level [0.1 - 1.0]')
        parser.add_argument('--alpha', default='0.5', type=float,
                            help='alpha')
        parser.add_argument('--beta', default='0.3', type=float,
                            help='beta')
        parser.add_argument('--gamma', default='0.1', type=float,
                            help='gamma')
        parser.add_argument('--batch', default=16, type=int,
                            help='batch size')
        parser.add_argument('--freq', default=20, type=int,
                            help='display frequency')
        parser.add_argument('--attack_search', action='store_true',
                            help='enable attack search')
        parser.add_argument('--epochs', default='20', type=int,
                            help='number of epochs')
        parser.add_argument('--model_name', default='', type=str,
                            help='model name ')
        parser.add_argument('--k', default=1, type=float,
                            help='disturbed parameter ')
        parser.add_argument('--nrpn', type=int, default=3, help='number of multi-rpn ')
        parser.add_argument('--z_size', type=int, default=128,
                            help='interpolate template size')
        parser.add_argument('--search_attack', action='store_true', help='set attacking in search region')
        self.isTrain = True
        return parser
