# Object Overlay

从视频中提取同一对象在不同时刻的位置，并合成为一张叠加图。工具不限定对象类别；仓库中的无人机只是一个使用样例。

![无人机多时刻叠加效果](./photo.jpg)

## 安装与部署

uv 支持 Windows、Linux 和 macOS。项目的使用方式相同，只有 uv 的安装命令因系统而异。

### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

也可以使用 Windows 包管理器：

```powershell
winget install --id=astral-sh.uv -e
```

### Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装完成后，进入项目目录并同步环境：

```console
uv sync
```

`uv` 会根据 `.python-version` 和 `uv.lock` 管理 Python、虚拟环境及项目依赖，不需要手动创建或激活虚拟环境。程序优先使用系统 FFmpeg；未安装时会使用项目依赖提供的 FFmpeg。其他安装方式见 [uv 官方安装文档](https://docs.astral.sh/uv/getting-started/installation/)。

## 使用

把自己的视频放在任意位置，然后运行：

```console
uv run object-overlay "path/to/input.mp4"
```

会先弹出一份设置菜单，用方向键选、Enter 确认，确认后才开始处理：

```text
? 要生成哪种叠加图
> overlay   只出无损叠加图
  ghost     只出残影版
  both      两个都出
```

菜单按键：

| 按键 | 功能 |
| --- | --- |
| ↑ / ↓ | 上下移动选项 |
| Enter 或 → | 确认当前选项，进入下一题 |
| ← 或 Esc | 返回上一题；在第一题时取消整个流程 |

依次会问：**要生成哪种叠加图 → 输出目录 → 轨迹线样式 → 轨迹线颜色 → 抽帧间隔 → 是否逐帧复核识别结果**。先定产物、再定处理。

只问当前设置下用得上的问题——比如输出内容选了 `overlay`，后两题里的轨迹线部分就不会出现。

适合自动识别的素材应当使用固定或基本固定的镜头，并且画面中有一个主要运动对象。`--interval` 越小，提供的候选位置越密。

自动识别有时会把不是目标的东西也抠进来，或者 Mask 切不准。最后一题选"是"（等价于加 `--select`）后会逐帧弹出复核窗口：

| 按键 | 功能 |
| --- | --- |
| Enter 或空格 | 纳入当前对象 |
| `S` | 跳过当前对象 |
| `B` 或 Backspace | 回退上一帧并重新选择 |
| `M` | 自动 Mask 不准确时，手动框选并修正 |
| `R` | 放弃当前人工修正，恢复自动 Mask |
| `Q` 或 Esc | 取消本次选择 |

当前候选对象显示为绿色；已经纳入的对象会以原始颜色留在后续预览中。全部候选处理完毕后，按 Enter 或空格生成结果。

### 只用参数

**命令行参数和菜单是叠加的，不是二选一**：参数先填进去当初始值，菜单在这上面改。喜欢直接用参数的开发者可以加 `-y`（或 `--yes`）跳过菜单：

```console
uv run object-overlay "path/to/input.mp4" --interval 1.0 --mode ghost \
  --trajectory-color "#00FF00" --select -y --output-dir "path/to/input_overlay"
```

在脚本、重定向、CI 这类没有终端的环境里，菜单会自动跳过，直接按参数跑，不加 `-y` 也不会卡住。

叠加图有两种，用 `--mode` 选择生成哪一种：`overlay` 只出无损版，`ghost` 只出残影版，`both` 两个都出（默认）。

- `overlay.png`：无损版本。Mask 内的像素直接从对应时刻的原帧复制，不做缩放和混合；Mask 之外的背景与背景帧逐像素相同。
- `overlay_ghost.png`：残影版本。各时刻按时间混合，越早越淡、越细，最新时刻最实、最粗，并用一条样条曲线把各时刻的位置连起来，用来表达"同一对象在时间上的多次曝光"而不是"多个对象"。

残影版的四个参数见下面的[参数表](#参数)。其中 `--no-trajectory` 只去掉那条线，渐隐和背景帧目标的擦除都保留。

轨迹线默认带一条深色描边：浅色线条在浅色天空上会融掉，描边保证任何背景下都读得出来。代价是浅色背景上会多出一道深边，不喜欢就用 `--no-trajectory-outline` 去掉。

```console
uv run object-overlay "path/to/input.mp4" --mode ghost --no-trajectory -y --output-dir "path/to/input_overlay"
```

背景帧（`--base-frame`）通常被直接当作画布，但它里面往往也有目标。残影版会先把背景帧里的目标用其余时刻在同一区域的**时域中值**擦掉，再把它当作序列中的**第一个时刻**参与合成——否则它会以完整不透明留在画面里，成为唯一一个脱离轨迹线的目标。检测不到背景帧目标时会跳过这一步；无损版不受影响，仍然使用原始背景帧。

抽帧、Mask 和元数据属于可复用的运行产物；每次执行 `--select` 都会重新选择要纳入的时刻。**是否做过人工筛选记在 `metadata.json` 的 `curated` 字段里**，不再体现在产物文件名上——产物身份由输出目录承担。

残影版的轨迹是否规整，取决于各时刻在时间上是否均匀。如果跳过大量候选帧，相邻时刻的间隔会忽大忽小，轨迹的弯折也会随之变得不规则；与其用 `--interval 0.5` 再手动跳掉一半，不如直接用 `--interval 1.0` 之类的粗间隔控制密度。

## 参数

交互式菜单里能改的都对应一个命令行参数，两者取值完全一致。

| 参数 | 取值 | 默认 | 说明 |
| --- | --- | --- | --- |
| `video` | 路径 | 必填 | 输入视频 |
| `-y`, `--yes` | 开关 | 关 | 跳过交互式设置，直接按当前参数开始 |
| `--mode` | `overlay` `ghost` `both` | `both` | 生成哪种叠加图 |
| `--interval` | 秒 | `0.5` | 抽帧时间间隔，决定候选位置的密度 |
| `--base-frame` | 整数 | `1` | 作为固定背景的抽帧编号 |
| `--output-dir` | 路径 | 见下方说明 | 结果目录 |
| `--force` | 开关 | 关 | 忽略已有缓存，重新生成全部 Mask |
| `--select` | 开关 | 关 | 逐帧复核识别结果：剔除认错的对象，Mask 不准的可手动修正 |
| `--ghost-min-alpha` | `(0, 1]` | `0.15` | 残影版中最早时刻的不透明度下限 |
| `--trajectory` / `--no-trajectory` | 开关 | 绘制 | 是否绘制轨迹线 |
| `--trajectory-outline` / `--no-trajectory-outline` | 开关 | 加 | 轨迹线下方是否加一条深色描边 |
| `--trajectory-color` | `#RRGGBB` | `#FFA500` | 轨迹线颜色，也接受 `#RGB` 简写 |

```console
uv run object-overlay --help
```

`--mode` 不是必给项，但一旦给了旧写法 `--mode motion` 会直接报错——那个取值在当前版本里没有含义。

### 输出目录

菜单第 2 题会问输出目录，两个选项：**默认**（选项里直接显示算出来的完整路径）或**自定义**（自己填一条）。命令行给了 `--output-dir` 就等价于预选"自定义"。

"默认"按下面的规则算：

- 视频放在名为 `input` 的目录里（本仓库样例就是这种布局），输出到它的**兄弟目录 `output`** 下：`input/drone.mp4` → `output/drone_overlay/`
- 其他情况放在**视频旁边**：`videos/drone.mp4` → `videos/drone_overlay/`

目录名都带视频名，所以多个视频的输出堆在同一个 `output/` 里也不会互相覆盖。

`examples/drone/input/` 和 `examples/drone/output/` 里各有一个 `.gitkeep` 占位。Git 不跟踪空目录，没有占位文件的话克隆下来这两个目录都不存在——有了它，把视频放进 `input/` 直接跑，结果就落在 `output/` 里。

## 无人机样例

本地将视频放入 `examples\drone\input\` 后，可运行：

```console
uv run object-overlay "./examples/drone/input/drone_1.mp4" --interval 1.0 --select -y --output-dir "./examples/drone/output/drone_1_selected"
```

这里的 `-y` 是跳过设置菜单——`--select` 本身已经有一套筛选界面，不必连着交互两次。想要设置菜单就把它去掉。

样例的输入视频和中间产物已由 `.gitignore` 忽略，仓库只保留最终示例图。

查看所有参数：

```console
uv run object-overlay --help
```

## 项目结构

```text
src/object_overlay/   核心代码
examples/drone/       无人机样例
tests/                核心功能测试
```
