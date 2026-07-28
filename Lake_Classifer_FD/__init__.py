def classFactory(iface):
    from .lake_classifier import LakeClassifierPlugin
    return LakeClassifierPlugin(iface)
