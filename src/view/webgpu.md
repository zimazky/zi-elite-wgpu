Precompute pipelines
  pipelines[]
  

Rendering pipelines
  pipelines[]
    shader
    textures
    uniform buffers
    vertex buffers
    targets

1. По умолчанию работает 1 конвейер, рисуется в канвас, сетка - квад
2. Должна быть функция добавления конвейера, по умолчанию: сетка - квад, без буфера глубины и шаблона.
   При создании определяем шейдер, разрешение, формат и количество целевых текстур.
3. Должна быть функция определения сетки для конвейера. Вертексный буфер может содержать множество сеток объектов.
4. Механизм отрисовки инстансов.
5. Механизм отрисовки последовательности разных объектов из буфера вершин.


```
pipeline
  vertex - vertex state
    module - shader module
    entryPoint - function in shader
    buffers - array of vertex buffer layouts
      arrayStride - stride in bytes between elenents of buffer array
      attributes - array defining the layout of vertex attributes
        shaderLocation - number of location
        format - type of attribute
        offset - offset in bytes from the beginning of element
  fragment - fragment state
    module - shader module
    entryPoint - function in shader
    targets - array of color target states (may multiple render targets)
      format - color target
  primitive - primitive state
    topology - points, lines, triangles in variations
  depthStencil
    format - "depth24plus"
    depthWriteEnabled
    depthCompare - "less"


  layout - pipeline layout


buffer descriptor
  size - количество элементов вершинного буфера
  usage - битовая маска для определения использования буфера (вершинный или индексный и доступ)
  mappedAtCreation - если true, то создается буфер отображаемом на локальную память состоянии
  
texture descriptor
  size
    width
    height
  format - формат цвета текстуры
  uasge - битовая маска определяющая использование текстуры
      
?texture view descriptor
  format - формат цвета текстуры
  dimension - размерность текстуры
  aspect - 'all', 'depth-only', 'ctencil-only'
  baseMipLevel - первый, самый детальный уровень mip
  mipLevelCount - число уровней mip
  ?baseArrayLayer - индекс первого слоя массива
  ?arrayLayerCount - число слоев

sampler descriptor
  addressModeU - тип выборки за краем текстуры в направлении U: 'repeat', 'clamp-to-edge', 'mirror-repeat'
  addressModeV - тип выборки за краем текстуры в направлении V
  magFilter - фильтрация: 'linear', 'nearest'
  minFilter - фильтрация
  mipmapFilter - выбор mip уровня: 'nearest', 'linear'
  maxAnisotropy: - уровень анизотропной фильтрации: 1-16
  


pipeline layout descriptor
  bindingGroupLayouts - array of binding group layouts


binding group layout descriptor
  entries - array
    binding - number of binding
    visibility - bitset of stages: vertex, fragment, compute
    buffer: {} - indicate type of binding
    texture: {} - indicate type of binding
    sampler: {} - indicate type of binding


binding group descriptor
  ?layout - 
  entries - array
    binding - number of binding
    resource - may be: sampler, texture view, external texture, buffer binding (uniform)
      buffer
      offset
      size


renderpass descriptor
  colorAttachments - массив
    view - texture view
    clearValue - цвет очищения 
     r
     g
     b
     a
    loadOp - начальная операция: 'clear', 'load'
    storeOp - операция сохранения: 'store', 'discard'
  depthStencilAttachment
    view: depthTexture.createView(),
    depthClearValue: 1.0,
    depthLoadOp: 'clear',
    depthStoreOp: "store",
    /*stencilClearValue: 0,
    stencilStoreOp: "store",
    depthLoadOp: 'clear',
    stencilLoadOp: 'clear'*/


```
