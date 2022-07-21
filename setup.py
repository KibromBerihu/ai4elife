from setuptools import setup, find_namespace_packages

setup(name='ai4elife',
      packages=find_namespace_packages(include=["ai4elife", "ai4elife.*"]),
      version='1.0.0',
      description='ai4elife, Data-centric aI framework for tumor segmentation.',
      url="https://github.com/KibromBerihu/ai4elife",
      author="LITO laboratory, institute Curie",
      author_email='kibrom.girum@curie.fr',
      license="MIT",
      #install_requires=[],
      keywords=['artificial intelligence', 'data-centric ai', 'medical image analysis',
                'lfbnet', 'FDG-PET', 'tumor segmentation', 'biomarkers']
      )
